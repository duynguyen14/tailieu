from __future__ import annotations

import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.core.config import DevicePreference, Settings
from app.domain.types import RenderedPage
from app.schemas.production_extraction import (
    ProductionExtractionResponse,
    ProductionModelStatus,
    ProductionPageResult,
    ProductionReadinessResponse,
    ProductionTiming,
)
from app.schemas.student_extraction import (
    STUDENT_APPLICANT_KEYS,
    STUDENT_NOTE_KEYS,
    build_student_user_prompt,
    student_target_to_compact_document,
)
from app.services.exceptions import ProductionExtractionError
from app.services.pdf_renderer import PdfRenderer
from app.services.text_reconstruction import (
    PageReconstructionInput,
    ReconstructionToken,
    reconstruct_case,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.models.onnx_page_classifier import OnnxPageClassifier
    from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine
    from app.models.paddle_orientation import PaddleOrientationClassifier


@dataclass
class _PreparedPage:
    page_number: int
    path: Path
    image_bytes: bytes
    width: int
    height: int
    rotation_angle: int
    orientation_confidence: float


@dataclass
class _OcrPage:
    prepared: _PreparedPage
    page_type: str
    classification_confidence: float
    tokens: list[OcrTokenPrediction]
    processing_time_ms: float


class _StudentRuntime:
    def __init__(self, settings: Settings) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = self._resolve_device(settings)
        precision = settings.production_student_precision
        if self.device == "cpu" or precision == "fp32":
            dtype = torch.float32
        elif precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 was requested but this GPU does not support BF16.")
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model_dir = settings.production_student_model_path
        if not (model_dir / "config.json").is_file():
            raise RuntimeError(f"Merged Gemma model was not found: {model_dir}")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        load_kwargs: dict[str, Any] = {"local_files_only": True}
        major = int(str(transformers.__version__).split(".", maxsplit=1)[0])
        load_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        self.stop_token_ids = self._generation_stop_token_ids()
        self.max_input_tokens = settings.production_student_max_input_tokens
        self.max_new_tokens = settings.production_student_max_new_tokens

    def _resolve_device(self, settings: Settings) -> str:
        torch = self.torch
        gpu_id = settings.production_gpu_device_id
        gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > gpu_id
        if gpu_available:
            return f"cuda:{gpu_id}"
        if settings.production_require_gpu:
            raise RuntimeError(
                f"Production extraction requires CUDA GPU {gpu_id}, but it is unavailable."
            )
        return "cpu"

    def _generation_stop_token_ids(self) -> list[int]:
        candidates: list[Any] = [
            getattr(self.model.generation_config, "eos_token_id", None),
            getattr(self.model.config, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
        values: list[int] = []
        for candidate in candidates:
            items = candidate if isinstance(candidate, (list, tuple)) else [candidate]
            for item in items:
                if isinstance(item, int) and item >= 0 and item not in values:
                    values.append(item)
        if not values:
            raise RuntimeError("Gemma exposes no generation stop token.")
        return values

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _normalize_target(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Gemma response root must be a JSON object.")
        note = value.get("n")
        applicants = value.get("a")
        if not isinstance(note, dict):
            raise ValueError("Gemma response n must be a JSON object.")
        if not isinstance(applicants, list):
            raise ValueError("Gemma response a must be a JSON array.")
        normalized_applicants: list[dict[str, Any]] = []
        for index, row in enumerate(applicants):
            if not isinstance(row, dict):
                raise ValueError(f"Gemma response a[{index}] must be a JSON object.")
            normalized_applicants.append(
                {alias: row.get(alias) for alias in STUDENT_APPLICANT_KEYS.values()}
            )
        return {
            "n": {alias: note.get(alias) for alias in STUDENT_NOTE_KEYS.values()},
            "a": normalized_applicants,
        }

    def warmup(self) -> None:
        # The warm-up intentionally generates one token.  It initializes CUDA,
        # attention kernels and the KV cache without spending seconds on a fake
        # full extraction.
        self.generate("Warm-up only.", max_new_tokens=1, validate=False)

    def generate(
        self,
        text: str,
        *,
        max_new_tokens: int | None = None,
        validate: bool = True,
    ) -> tuple[dict[str, Any], int, float]:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": build_student_user_prompt(text)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > self.max_input_tokens:
            raise ProductionExtractionError(
                f"Reconstructed text produces {prompt_tokens} prompt tokens; "
                f"maximum is {self.max_input_tokens}."
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.stop_token_ids,
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        output_ids = generated[0, prompt_tokens:]
        generated_tokens = int(output_ids.shape[-1])
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if not validate:
            return {}, generated_tokens, latency_ms
        try:
            parsed = json.loads(self._strip_markdown_fence(raw))
            return self._normalize_target(parsed), generated_tokens, latency_ms
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProductionExtractionError(
                f"Gemma returned invalid compact JSON: {exc}"
            ) from exc


class ProductionExtractionService:
    """Preloaded, single-flight PDF -> validated business JSON pipeline.

    Seal detection is intentionally absent.  The service does not construct a
    Paddle layout detector and never removes OCR tokens based on seals.
    """

    def __init__(self, settings: Settings, renderer: PdfRenderer) -> None:
        self.settings = settings
        self.renderer = renderer
        self._lock = threading.Lock()
        self._loading = False
        self._ready = False
        self._error: str | None = None
        self._classifier: OnnxPageClassifier | None = None
        self._orientation: PaddleOrientationClassifier | None = None
        self._ocr: PaddleOcrEngine | None = None
        self._student: _StudentRuntime | None = None
        self._models: list[ProductionModelStatus] = []

    def initialize(self) -> None:
        if not self.settings.production_extraction_enabled or self._ready:
            return
        self._loading = True
        self._error = None
        try:
            from app.models.onnx_page_classifier import OnnxPageClassifier
            from app.models.paddle_ocr_engine import PaddleOcrEngine
            from app.models.paddle_orientation import PaddleOrientationClassifier

            if (
                self.settings.production_require_gpu
                and self.settings.ocr_runtime_gpu_device_id
                != self.settings.production_gpu_device_id
            ):
                raise RuntimeError(
                    "OCR_RUNTIME_GPU_DEVICE_ID and PRODUCTION_GPU_DEVICE_ID must match "
                    "so every production model is preloaded on the same GPU."
                )
            model_path = self.settings.page_classifier_model_resolved_path
            labels_path = self.settings.page_classifier_labels_resolved_path
            if model_path is None or labels_path is None:
                raise RuntimeError(
                    "PAGE_CLASSIFIER_MODEL_PATH and PAGE_CLASSIFIER_LABELS_PATH are required."
                )
            self._classifier = OnnxPageClassifier(
                model_path=model_path,
                labels_path=labels_path,
                device=DevicePreference.CUDA,
                gpu_device_id=self.settings.production_gpu_device_id,
                allow_cpu_fallback=not self.settings.production_require_gpu,
            )
            self._ocr = PaddleOcrEngine(
                self.settings,
                device_override=DevicePreference.CUDA,
            )
            self._orientation = PaddleOrientationClassifier(
                self.settings,
                device=self._ocr.effective_device,
            )
            self._student = _StudentRuntime(self.settings)
            devices = {
                self._classifier.effective_device,
                self._ocr.effective_device,
                self._orientation.effective_device,
                self._student.device,
            }
            if self.settings.production_require_gpu and any(
                device == "cpu" for device in devices
            ):
                raise RuntimeError(f"A production model fell back to CPU: {sorted(devices)}")
            self._models = [
                ProductionModelStatus(
                    name="page_classifier",
                    loaded=True,
                    device=self._classifier.effective_device,
                    model=str(model_path),
                ),
                ProductionModelStatus(
                    name="orientation",
                    loaded=True,
                    device=self._orientation.effective_device,
                    model=self._orientation.model_name,
                ),
                ProductionModelStatus(
                    name="ocr_detection_recognition",
                    loaded=True,
                    device=self._ocr.effective_device,
                    model=self._ocr.model_name,
                    message="Seal/layout exclusion model is not loaded.",
                ),
                ProductionModelStatus(
                    name="student_extractor",
                    loaded=True,
                    device=self._student.device,
                    model=str(self._student.model_dir),
                ),
            ]
            if self.settings.production_warmup_models:
                self._warmup()
            self._ready = True
            LOGGER.info("Production extraction runtime is ready; devices=%s", sorted(devices))
        except Exception as exc:
            self._error = str(exc)
            self._models = []
            raise
        finally:
            self._loading = False

    def _warmup(self) -> None:
        from PIL import Image

        assert self._classifier is not None
        assert self._orientation is not None
        assert self._ocr is not None
        assert self._student is not None
        with tempfile.TemporaryDirectory(prefix="production-warmup-") as temp_dir:
            path = Path(temp_dir) / "warmup.png"
            image = Image.new("RGB", (640, 640), "white")
            image.save(path, format="PNG")
            image_bytes = path.read_bytes()
            self._classifier.predict_bytes([image_bytes], top_k=1)
            self._orientation.predict(path)
            self._ocr.predict(path)
            image.close()
        self._student.warmup()

    def readiness(self) -> ProductionReadinessResponse:
        return ProductionReadinessResponse(
            enabled=self.settings.production_extraction_enabled,
            ready=self._ready,
            loading=self._loading,
            seal_detection_enabled=False,
            models=self._models,
            error=self._error,
        )

    def close(self) -> None:
        # Framework runtimes own their native resources.  Dropping references
        # lets process shutdown release CUDA contexts without unsafe ad-hoc
        # destruction while another native finalizer may still be running.
        self._ready = False

    def extract(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionExtractionResponse:
        if not self._ready:
            raise ProductionExtractionError(
                self._error or "Production models are not loaded yet."
            )
        if not self._lock.acquire(blocking=False):
            raise ProductionExtractionError(
                "The GPU extraction worker is busy. Retry this request shortly."
            )
        try:
            return self._extract_locked(pdf_path=pdf_path, filename=filename, sha256=sha256)
        finally:
            self._lock.release()

    def _extract_locked(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionExtractionResponse:
        assert self._classifier is not None
        assert self._orientation is not None
        assert self._ocr is not None
        assert self._student is not None
        request_id = str(uuid4())
        total_started = time.perf_counter()
        warnings: list[str] = []

        render_started = time.perf_counter()
        rendered = self.renderer.render(pdf_path)
        render_ms = (time.perf_counter() - render_started) * 1000

        with tempfile.TemporaryDirectory(prefix="production-extraction-") as temp_dir:
            root = Path(temp_dir)
            orientation_started = time.perf_counter()
            prepared_pages = self._orient_pages(rendered, root)
            orientation_ms = (time.perf_counter() - orientation_started) * 1000

            classification_started = time.perf_counter()
            classifier_pages = [
                RenderedPage(
                    page_index=page.page_number - 1,
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    image_bytes=page.image_bytes,
                )
                for page in prepared_pages
            ]
            predictions = self._classifier.predict_bytes(
                [page.image_bytes for page in classifier_pages], top_k=1
            )
            classification_ms = (time.perf_counter() - classification_started) * 1000

            ocr_started = time.perf_counter()
            ocr_pages: list[_OcrPage] = []
            target_types = set(self.settings.production_ocr_page_type_list)
            page_results: list[ProductionPageResult] = []
            for page, prediction in zip(prepared_pages, predictions, strict=True):
                page_type = prediction.label.value
                use_page = page_type in target_types
                tokens: list[OcrTokenPrediction] = []
                page_ocr_ms = 0.0
                if use_page:
                    page_ocr_started = time.perf_counter()
                    tokens = self._recognize_page(page.path, root / f"ocr-{page.page_number}")
                    page_ocr_ms = (time.perf_counter() - page_ocr_started) * 1000
                    ocr_pages.append(
                        _OcrPage(
                            prepared=page,
                            page_type=page_type,
                            classification_confidence=prediction.confidence,
                            tokens=tokens,
                            processing_time_ms=page_ocr_ms,
                        )
                    )
                if prediction.confidence < self.settings.low_confidence_threshold:
                    warnings.append(
                        f"Page {page.page_number} classification confidence "
                        f"{prediction.confidence:.3f} is below "
                        f"{self.settings.low_confidence_threshold:.3f}."
                    )
                page_results.append(
                    ProductionPageResult(
                        page_number=page.page_number,
                        rotation_angle=page.rotation_angle,
                        orientation_confidence=page.orientation_confidence,
                        page_type=page_type,
                        classification_confidence=prediction.confidence,
                        used_for_extraction=use_page,
                        ocr_token_count=len(tokens),
                        ocr_processing_time_ms=round(page_ocr_ms, 3),
                    )
                )
            ocr_ms = (time.perf_counter() - ocr_started) * 1000
            if not ocr_pages:
                raise ProductionExtractionError(
                    "No page was classified as an OCR/extraction target."
                )
            if not any(page.tokens for page in ocr_pages):
                raise ProductionExtractionError("OCR returned no usable text tokens.")

            reconstruction_started = time.perf_counter()
            reconstruction_inputs = [
                PageReconstructionInput(
                    page_id=f"page_{page.prepared.page_number}",
                    page_number=page.prepared.page_number,
                    page_type=page.page_type,
                    page_role="NOTE",
                    tokens=tuple(
                        ReconstructionToken(
                            id=f"p{page.prepared.page_number}_{token.token_id}",
                            text=token.text,
                            bbox=token.bbox,
                            order=token.token_index,
                        )
                        for token in page.tokens
                    ),
                )
                for page in ocr_pages
            ]
            reconstructed_text, reconstruction_results = reconstruct_case(
                reconstruction_inputs,
                tolerance_ratio=self.settings.text_reconstruction_line_tolerance,
            )
            reconstruction_ms = (time.perf_counter() - reconstruction_started) * 1000
            for result in reconstruction_results:
                warnings.extend(result.warnings)

            compact_output, _generated_tokens, extraction_ms = self._student.generate(
                reconstructed_text
            )
            extraction = student_target_to_compact_document(
                compact_output,
                case_id=request_id,
                document_id=request_id,
            )

        total_ms = (time.perf_counter() - total_started) * 1000
        return ProductionExtractionResponse(
            request_id=request_id,
            filename=filename,
            sha256=sha256,
            page_count=len(rendered),
            extraction=extraction,
            compact_model_output=compact_output,
            reconstructed_text=(
                reconstructed_text
                if self.settings.production_include_reconstructed_text
                else None
            ),
            pages=page_results,
            timings=ProductionTiming(
                render_ms=round(render_ms, 3),
                orientation_ms=round(orientation_ms, 3),
                classification_ms=round(classification_ms, 3),
                ocr_ms=round(ocr_ms, 3),
                reconstruction_ms=round(reconstruction_ms, 3),
                extraction_ms=round(extraction_ms, 3),
                total_ms=round(total_ms, 3),
            ),
            warnings=list(dict.fromkeys(warnings)),
        )

    def _orient_pages(
        self, pages: list[RenderedPage], root: Path
    ) -> list[_PreparedPage]:
        from app.models.ocr_preprocessing import (
            load_rgb_image,
            orientation_probe,
            png_bytes,
        )

        assert self._orientation is not None
        output: list[_PreparedPage] = []
        for page in pages:
            source_path = root / f"source-{page.page_number}.png"
            source_path.write_bytes(page.image_bytes)
            source = load_rgb_image(source_path)
            probe, _detection = orientation_probe(
                source,
                thumbnail_max_side=self.settings.ocr_content_thumbnail_max_side,
                padding_ratio=self.settings.ocr_content_padding_ratio,
                min_components=self.settings.ocr_content_min_components,
                min_content_area_ratio=self.settings.ocr_content_min_area_ratio,
                max_crop_area_ratio=self.settings.ocr_content_max_crop_area_ratio,
            )
            prediction = self._orientation.predict_image(probe)
            probe.close()
            rotated = (
                source.rotate(prediction.angle, expand=True, fillcolor="white")
                if prediction.angle
                else source.copy()
            )
            path = root / f"oriented-{page.page_number}.png"
            image_bytes = png_bytes(rotated)
            path.write_bytes(image_bytes)
            width, height = rotated.size
            rotated.close()
            source.close()
            output.append(
                _PreparedPage(
                    page_number=page.page_number,
                    path=path,
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    rotation_angle=prediction.angle,
                    orientation_confidence=prediction.confidence,
                )
            )
        return output

    def _recognize_page(self, image_path: Path, temp_root: Path) -> list[OcrTokenPrediction]:
        from app.models.ocr_preprocessing import (
            load_rgb_image,
            ocr_quality_score,
            png_bytes,
            prepare_ocr_image,
            recommended_upscale,
        )
        from app.models.ocr_region_refinement import refine_ocr_tokens

        assert self._ocr is not None
        temp_root.mkdir(parents=True, exist_ok=True)
        source = load_rgb_image(image_path)
        prepare_kwargs = {
            "rotation": 0,
            "enable_content_crop": self.settings.ocr_enable_content_crop,
            "thumbnail_max_side": self.settings.ocr_content_thumbnail_max_side,
            "padding_ratio": self.settings.ocr_content_padding_ratio,
            "min_components": self.settings.ocr_content_min_components,
            "min_content_area_ratio": self.settings.ocr_content_min_area_ratio,
            "max_crop_area_ratio": self.settings.ocr_content_max_crop_area_ratio,
            "max_processed_side": self.settings.ocr_upscale_max_processed_side,
        }
        selected = prepare_ocr_image(source, scale=1.0, **prepare_kwargs)
        selected_path = temp_root / "base.png"
        selected_path.write_bytes(png_bytes(selected.image))
        tokens = self._ocr.predict(selected_path)
        selected_score = ocr_quality_score(tokens)
        if self.settings.ocr_enable_small_text_upscale:
            scale = recommended_upscale(
                tokens,
                target_text_height=self.settings.ocr_upscale_target_text_height,
                max_factor=self.settings.ocr_upscale_max_factor,
            )
            if scale > 1.05:
                upscaled = prepare_ocr_image(source, scale=scale, **prepare_kwargs)
                upscaled_path = temp_root / "upscaled.png"
                upscaled_path.write_bytes(png_bytes(upscaled.image))
                upscaled_tokens = self._ocr.predict(upscaled_path)
                if ocr_quality_score(upscaled_tokens) >= selected_score - 0.05:
                    selected.image.close()
                    selected = upscaled
                    tokens = upscaled_tokens
                else:
                    upscaled.image.close()
        if self.settings.production_ocr_enable_refinement and tokens:
            refinement = refine_ocr_tokens(
                selected.image,
                tokens,
                temp_dir=temp_root / "refinement",
                predict_many=self._ocr.predict_refinement_many,
                scales=self.settings.ocr_refinement_scale_list,
                confidence_threshold=self.settings.ocr_refinement_confidence_threshold,
                candidate_min_confidence=(
                    self.settings.ocr_refinement_candidate_min_confidence
                ),
                agreement_similarity=self.settings.ocr_refinement_agreement_similarity,
                min_confidence_gain=self.settings.ocr_refinement_min_confidence_gain,
                max_regions=self.settings.ocr_refinement_max_regions,
                max_seconds=self.settings.ocr_refinement_max_seconds,
                padding_ratio=self.settings.ocr_refinement_padding_ratio,
                max_crop_side=self.settings.ocr_refinement_max_crop_side,
                protect_digits=self.settings.ocr_refinement_protect_digits,
                version=self.settings.ocr_refinement_version,
            )
            tokens = refinement.tokens
        selected.image.close()
        source.close()
        return tokens

