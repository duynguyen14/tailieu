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

import httpx

from app.core.config import DevicePreference, Settings
from app.domain.types import RenderedPage
from app.schemas.production_extraction import (
    ProductionExtractionResponse,
    ProductionModelStatus,
    ProductionOcrResponse,
    ProductionPageClassificationResponse,
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
from app.services.student_model_runtime import (
    apply_student_chat_template,
    generation_stop_token_ids,
    parse_student_json_output,
)
from app.services.text_reconstruction import (
    PageReconstructionInput,
    ReconstructionToken,
    reconstruct_case,
)

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL import Image

    from app.models.ocr_preprocessing import ContentDetection
    from app.models.onnx_page_classifier import OnnxPageClassifier
    from app.models.paddle_ocr_engine import OcrTokenPrediction, PaddleOcrEngine
    from app.models.paddle_orientation import PaddleOrientationClassifier


@dataclass
class _PreparedPage:
    page_number: int
    image: Image.Image
    image_bytes: bytes | None
    width: int
    height: int
    rotation_angle: int
    orientation_confidence: float
    content_detection: ContentDetection | None = None

    def close(self) -> None:
        self.image.close()


@dataclass
class _OcrPage:
    prepared: _PreparedPage
    page_type: str
    classification_confidence: float
    tokens: list[OcrTokenPrediction]
    processing_time_ms: float


class _StudentRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.api_base_url = (
            settings.production_student_api_base_url.strip().rstrip("/")
            if settings.production_student_api_base_url
            else None
        )
        self.max_input_tokens = settings.production_student_max_input_tokens
        self.max_new_tokens = settings.production_student_max_new_tokens
        self._api_client: httpx.Client | None = None

        if self.api_base_url:
            self.device = f"api:{self.api_base_url}"
            self.model_dir = Path(settings.production_student_api_model)
            self._api_client = httpx.Client(
                timeout=settings.production_student_api_timeout_seconds
            )
            LOGGER.info(
                "Initialized remote student API runtime; base_url=%s model=%s",
                self.api_base_url,
                settings.production_student_api_model,
            )
            return

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
            raise RuntimeError(f"Merged student model was not found: {model_dir}")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        load_kwargs: dict[str, Any] = {"local_files_only": True}
        major = int(str(transformers.__version__).split(".", maxsplit=1)[0])
        load_kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
        attn_impl = "sdpa"
        if self.device.startswith("cuda"):
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                attn_implementation=attn_impl,
                **load_kwargs,
            )
        except (TypeError, ValueError, ImportError):
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
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        self.stop_token_ids = generation_stop_token_ids(self.tokenizer, self.model)
        self._eager_forward = self.model.forward
        self._supports_logits_to_keep = True

        # Pre-allocated StaticCache + explicit CUDA Graph state for zero-overhead
        # 1-token decoding on GPU.
        self._static_cache: Any | None = None
        self._cuda_graph: Any | None = None
        self._graph_input_ids: Any | None = None
        self._graph_cache_position: Any | None = None
        self._graph_next_token: Any | None = None
        self._max_cache_len = self.max_input_tokens + self.max_new_tokens + 64

        #Immutable prefix KV cache for the static instruction header preceding
        # DOCUMENT_TEXT so every request skips prefilling those ~200 tokens.
        self._prefix_ids: list[int] = []
        self._prefix_len: int = 0
        self._prefix_kv_snapshot: list[tuple[Any, Any]] | None = None
        self._dynamic_prefix_kv: list[tuple[Any, Any]] | None = None

        self.compiled = False
        if (
            self._cuda_graph is None
            and self.device.startswith("cuda")
            and settings.production_student_torch_compile
            and hasattr(torch, "compile")
        ):
            import sysconfig

            include_dir = Path(sysconfig.get_paths().get("include", ""))
            if not (include_dir / "Python.h").is_file():
                LOGGER.warning(
                    "Skipping torch.compile because Python development headers "
                    "(Python.h) are not installed in %s.",
                    include_dir,
                )
            else:
                try:
                    compiled_fn = torch.compile(
                        self._eager_forward,
                        mode=settings.production_student_torch_compile_mode,
                        fullgraph=False,
                        dynamic=True,
                    )

                    def _selective_forward(*args: Any, **kwargs: Any) -> Any:
                        input_ids = kwargs.get("input_ids")
                        if input_ids is None and args:
                            input_ids = args[0]
                        if input_ids is not None and input_ids.shape[-1] == 1:
                            return compiled_fn(*args, **kwargs)
                        return self._eager_forward(*args, **kwargs)

                    self.model.forward = _selective_forward
                    self.compiled = True
                    LOGGER.info(
                        "Enabled selective decode torch.compile for student runtime; mode=%s",
                        settings.production_student_torch_compile_mode,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Could not enable torch.compile for student runtime, using eager mode: %s",
                        exc,
                    )
                    self.model.forward = self._eager_forward
                    self.compiled = False

    def _reset_static_cache(self) -> None:
        if self._static_cache is None:
            return
        if hasattr(self._static_cache, "reset"):
            self._static_cache.reset()
            return
        for layer in getattr(self._static_cache, "layers", []):
            if hasattr(layer, "keys") and layer.keys is not None:
                layer.keys.zero_()
            if hasattr(layer, "values") and layer.values is not None:
                layer.values.zero_()
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length.zero_()

    def _snapshot_static_cache_prefix(self, length: int) -> list[tuple[Any, Any]] | None:
        if self._static_cache is None or length <= 0:
            return None
        layers = getattr(self._static_cache, "layers", None)
        if layers:
            snapshot: list[tuple[Any, Any]] = []
            for layer in layers:
                if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                    return None
                snapshot.append(
                    (
                        layer.keys[:, :, :length, :].clone(),
                        layer.values[:, :, :length, :].clone(),
                    )
                )
            return snapshot
        key_cache = getattr(self._static_cache, "key_cache", None)
        value_cache = getattr(self._static_cache, "value_cache", None)
        if key_cache and value_cache:
            return [
                (k[:, :, :length, :].clone(), v[:, :, :length, :].clone())
                for k, v in zip(key_cache, value_cache, strict=True)
            ]
        return None

    def _restore_static_cache_prefix(self) -> bool:
        if self._static_cache is None or not self._prefix_kv_snapshot or self._prefix_len <= 0:
            return False
        self._reset_static_cache()
        length = self._prefix_len
        layers = getattr(self._static_cache, "layers", None)
        if layers:
            for layer, (k_snap, v_snap) in zip(layers, self._prefix_kv_snapshot, strict=True):
                layer.keys[:, :, :length, :].copy_(k_snap)
                layer.values[:, :, :length, :].copy_(v_snap)
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length.fill_(length)
            return True
        key_cache = getattr(self._static_cache, "key_cache", None)
        value_cache = getattr(self._static_cache, "value_cache", None)
        if key_cache and value_cache:
            for k_buf, v_buf, (k_snap, v_snap) in zip(
                key_cache, value_cache, self._prefix_kv_snapshot, strict=True
            ):
                k_buf[:, :, :length, :].copy_(k_snap)
                v_buf[:, :, :length, :].copy_(v_snap)
            return True
        return False

    def _init_static_and_prefix_cache(
        self, *, enable_cuda_graph: bool, enable_prefix_cache: bool
    ) -> None:
        from transformers.cache_utils import StaticCache

        torch = self.torch
        with torch.cuda.device(self.device), torch.inference_mode():
            try:
                self._static_cache = StaticCache(
                    config=self.model.config,
                    max_batch_size=1,
                    max_cache_len=self._max_cache_len,
                    device=self.device,
                    dtype=self.model.dtype,
                )
                self._reset_static_cache()
                dummy_ids = torch.zeros((1, 8), dtype=torch.long, device=self.device)
                dummy_pos = torch.arange(0, 8, dtype=torch.long, device=self.device)
                try:
                    self._eager_forward(
                        input_ids=dummy_ids,
                        past_key_values=self._static_cache,
                        cache_position=dummy_pos,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                except TypeError:
                    self._supports_logits_to_keep = False
                    self._eager_forward(
                        input_ids=dummy_ids,
                        past_key_values=self._static_cache,
                        cache_position=dummy_pos,
                        use_cache=True,
                        return_dict=True,
                    )
            except Exception as exc:
                LOGGER.warning("Could not initialize StaticCache (%s).", exc)
                self._static_cache = None

            if enable_cuda_graph and self._static_cache is not None:
                try:
                    self._graph_input_ids = torch.zeros(
                        (1, 1), dtype=torch.long, device=self.device
                    )
                    self._graph_cache_position = torch.zeros(
                        (1,), dtype=torch.long, device=self.device
                    )
                    step_extra = (
                        {"logits_to_keep": 1} if self._supports_logits_to_keep else {}
                    )
                    capture_stream = torch.cuda.Stream(device=self.device)
                    capture_stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(capture_stream):
                        for pos in (8, 9, 10):
                            self._graph_cache_position.fill_(pos)
                            out = self._eager_forward(
                                input_ids=self._graph_input_ids,
                                past_key_values=self._static_cache,
                                cache_position=self._graph_cache_position,
                                position_ids=self._graph_cache_position.unsqueeze(0),
                                use_cache=True,
                                return_dict=True,
                                **step_extra,
                            )
                            self._graph_next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    torch.cuda.current_stream(self.device).wait_stream(capture_stream)
                    torch.cuda.synchronize(self.device)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        out = self._eager_forward(
                            input_ids=self._graph_input_ids,
                            past_key_values=self._static_cache,
                            cache_position=self._graph_cache_position,
                            position_ids=self._graph_cache_position.unsqueeze(0),
                            use_cache=True,
                            return_dict=True,
                            **step_extra,
                        )
                        self._graph_next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    torch.cuda.synchronize(self.device)
                    self._cuda_graph = graph
                except Exception as exc:
                    LOGGER.info(
                        "Manual CUDA Graph capture skipped (%s); using DynamicCache + compiled decode.",
                        exc,
                    )
                    self._cuda_graph = None
                    self._static_cache = None
            else:
                self._static_cache = None

            if enable_prefix_cache:
                try:
                    self._precompute_instruction_prefix_cache()
                except Exception as exc:
                    LOGGER.warning("Could not precompute prefix KV cache: %s", exc)

        LOGGER.info(
            "Initialized student decoder; device=%s static_cache=%s cuda_graph=%s prefix_cached_tokens=%d",
            self.device,
            self._static_cache is not None,
            self._cuda_graph is not None,
            self._prefix_len,
        )

    def _precompute_instruction_prefix_cache(self) -> None:
        torch = self.torch
        marker = "DOCUMENT_TEXT_MARKER_987654321"
        sample_prompt = apply_student_chat_template(
            self.tokenizer,
            [{"role": "user", "content": build_student_user_prompt(marker)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_text = sample_prompt.split(marker, maxsplit=1)[0]
        if not prefix_text:
            return
        encoded = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        prefix_ids_tensor = encoded["input_ids"].to(self.device)
        # Keep the last 2 tokens out of the frozen prefix so BPE boundary merges
        # with the first document character never alter tokenization.
        safe_len = int(prefix_ids_tensor.shape[-1]) - 2
        if safe_len <= 16:
            return
        safe_prefix_tensor = prefix_ids_tensor[:, :safe_len]
        step_extra = {"logits_to_keep": 1} if self._supports_logits_to_keep else {}

        if self._static_cache is not None:
            self._reset_static_cache()
            cache_pos = torch.arange(0, safe_len, dtype=torch.long, device=self.device)
            self._eager_forward(
                input_ids=safe_prefix_tensor,
                past_key_values=self._static_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
                **step_extra,
            )
            snapshot = self._snapshot_static_cache_prefix(safe_len)
            if snapshot is not None:
                self._prefix_ids = safe_prefix_tensor[0].tolist()
                self._prefix_len = safe_len
                self._prefix_kv_snapshot = snapshot

        # Also snapshot DynamicCache layers for non-CUDA-graph path
        dyn_out = self._eager_forward(
            input_ids=safe_prefix_tensor,
            use_cache=True,
            return_dict=True,
            **step_extra,
        )
        dyn_kv = dyn_out.past_key_values
        dyn_layers = getattr(dyn_kv, "layers", None)
        if dyn_layers and all(
            hasattr(layer, "keys") and hasattr(layer, "values") for layer in dyn_layers
        ):
            self._dynamic_prefix_kv = [
                (layer.keys.clone(), layer.values.clone()) for layer in dyn_layers
            ]
            self._prefix_ids = safe_prefix_tensor[0].tolist()
            self._prefix_len = safe_len
        elif hasattr(dyn_kv, "key_cache") and hasattr(dyn_kv, "value_cache"):
            self._dynamic_prefix_kv = [
                (k.clone(), v.clone())
                for k, v in zip(dyn_kv.key_cache, dyn_kv.value_cache, strict=True)
            ]
            self._prefix_ids = safe_prefix_tensor[0].tolist()
            self._prefix_len = safe_len

    def _build_dynamic_prefix_cache(self) -> Any | None:
        if not self._dynamic_prefix_kv or self._prefix_len <= 0:
            return None
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for idx, (k_snap, v_snap) in enumerate(self._dynamic_prefix_kv):
            cache.update(k_snap.clone(), v_snap.clone(), idx)
        return cache

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

    @staticmethod
    def _normalize_target(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("Student response root must be a JSON object.")
        note = value.get("n")
        applicants = value.get("a")
        if not isinstance(note, dict):
            raise ValueError("Student response n must be a JSON object.")
        if not isinstance(applicants, list):
            raise ValueError("Student response a must be a JSON array.")
        normalized_applicants: list[dict[str, Any]] = []
        for index, row in enumerate(applicants):
            if not isinstance(row, dict):
                raise ValueError(f"Student response a[{index}] must be a JSON object.")
            normalized_applicants.append(
                {alias: row.get(alias) for alias in STUDENT_APPLICANT_KEYS.values()}
            )
        return {
            "n": {alias: note.get(alias) for alias in STUDENT_NOTE_KEYS.values()},
            "a": normalized_applicants,
        }

    def warmup(self) -> None:
        if self.api_base_url:
            return
        warmup_tokens = 4 if (self.compiled or self._cuda_graph is not None) else 1
        self.generate("Warm-up only.", max_new_tokens=warmup_tokens, validate=False)

    def _static_cache_greedy_decode(
        self,
        input_ids: Any,
        *,
        limit: int,
    ) -> Any:
        torch = self.torch
        assert self._static_cache is not None

        stop_set = set(self.stop_token_ids)
        stop_tensor = torch.tensor(
            self.stop_token_ids, device=input_ids.device, dtype=input_ids.dtype
        )
        prompt_len = int(input_ids.shape[-1])
        step_extra = {"logits_to_keep": 1} if self._supports_logits_to_keep else {}

        # Use precomputed instruction KV cache if the prompt begins with the exact
        # cached prefix tokens (saves ~180 tokens of prefill compute every request).
        if (
            self._prefix_len > 0
            and prompt_len > self._prefix_len
            and input_ids[0, : self._prefix_len].tolist() == self._prefix_ids
            and self._restore_static_cache_prefix()
        ):
            suffix_ids = input_ids[:, self._prefix_len :]
            cache_pos = torch.arange(
                self._prefix_len, prompt_len, dtype=torch.long, device=self.device
            )
            prefill_out = self._eager_forward(
                input_ids=suffix_ids,
                past_key_values=self._static_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
                **step_extra,
            )
        else:
            self._reset_static_cache()
            cache_pos = torch.arange(0, prompt_len, dtype=torch.long, device=self.device)
            prefill_out = self._eager_forward(
                input_ids=input_ids,
                past_key_values=self._static_cache,
                cache_position=cache_pos,
                use_cache=True,
                return_dict=True,
                **step_extra,
            )

        next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1).clone()
        generated_tokens = [next_token]
        if limit <= 1:
            return next_token[0]

        use_graph = (
            self._cuda_graph is not None
            and self._graph_input_ids is not None
            and self._graph_cache_position is not None
            and self._graph_next_token is not None
        )
        cur_pos = prompt_len
        pos_tensor = torch.zeros((1,), dtype=torch.long, device=self.device)

        for step in range(1, limit):
            if cur_pos >= self._max_cache_len - 1:
                break
            if use_graph:
                self._graph_input_ids.copy_(next_token)
                self._graph_cache_position.fill_(cur_pos)
                self._cuda_graph.replay()
                next_token = self._graph_next_token.clone()
            else:
                pos_tensor.fill_(cur_pos)
                step_out = self.model.forward(
                    input_ids=next_token,
                    past_key_values=self._static_cache,
                    cache_position=pos_tensor,
                    position_ids=pos_tensor.unsqueeze(0),
                    use_cache=True,
                    return_dict=True,
                    **step_extra,
                )
                next_token = step_out.logits[:, -1:, :].argmax(dim=-1)
            generated_tokens.append(next_token)
            cur_pos += 1
            if (step & 15) == 0 or step == limit - 1:
                if torch.isin(torch.cat(generated_tokens[-16:], dim=-1), stop_tensor).any():
                    break

        all_tokens = torch.cat(generated_tokens, dim=-1)[0]
        token_list = all_tokens.tolist()
        for idx, tok in enumerate(token_list):
            if tok in stop_set:
                return all_tokens[: idx + 1]
        return all_tokens

    def _fast_greedy_decode(
        self,
        input_ids: Any,
        attention_mask: Any | None,
        *,
        limit: int,
    ) -> Any:
        if self._static_cache is not None and self._cuda_graph is not None:
            return self._static_cache_greedy_decode(input_ids, limit=limit)

        torch = self.torch
        stop_set = set(self.stop_token_ids)
        stop_tensor = torch.tensor(
            self.stop_token_ids, device=input_ids.device, dtype=input_ids.dtype
        )
        prompt_len = int(input_ids.shape[-1])
        step_extra: dict[str, Any] = (
            {"logits_to_keep": 1} if self._supports_logits_to_keep else {}
        )
        prefix_cache = (
            self._build_dynamic_prefix_cache()
            if (
                self._prefix_len > 0
                and prompt_len > self._prefix_len
                and input_ids[0, : self._prefix_len].tolist() == self._prefix_ids
            )
            else None
        )
        if prefix_cache is not None:
            suffix_ids = input_ids[:, self._prefix_len :]
            prefill_out = self._eager_forward(
                input_ids=suffix_ids,
                past_key_values=prefix_cache,
                use_cache=True,
                return_dict=True,
                **step_extra,
            )
        else:
            prefill_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "use_cache": True,
                "return_dict": True,
            }
            if attention_mask is not None:
                prefill_kwargs["attention_mask"] = attention_mask
            try:
                prefill_out = self._eager_forward(**prefill_kwargs, **step_extra)
            except TypeError:
                self._supports_logits_to_keep = False
                step_extra = {}
                prefill_out = self._eager_forward(**prefill_kwargs)

        past_key_values = prefill_out.past_key_values
        next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)
        generated_tokens = [next_token]
        if limit <= 1 or int(next_token.item()) in stop_set:
            return torch.cat(generated_tokens, dim=-1)[0]

        for step in range(1, limit):
            step_out = self.model.forward(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                **step_extra,
            )
            past_key_values = step_out.past_key_values
            next_token = step_out.logits[:, -1:, :].argmax(dim=-1)
            generated_tokens.append(next_token)
            if (step & 15) == 0 or step == limit - 1:
                if torch.isin(torch.cat(generated_tokens[-16:], dim=-1), stop_tensor).any():
                    break

        all_tokens = torch.cat(generated_tokens, dim=-1)[0]
        token_list = all_tokens.tolist()
        for idx, tok in enumerate(token_list):
            if tok in stop_set:
                return all_tokens[: idx + 1]
        return all_tokens

    def _generate_via_api(
        self,
        text: str,
        *,
        max_new_tokens: int | None = None,
        validate: bool = True,
    ) -> tuple[dict[str, Any], int, float]:
        assert self._api_client is not None
        assert self.api_base_url is not None
        headers: dict[str, str] = {}
        if self.settings.production_student_api_key is not None:
            key = self.settings.production_student_api_key.get_secret_value().strip()
            if key:
                headers["Authorization"] = f"Bearer {key}"
        payload: dict[str, Any] = {
            "model": self.settings.production_student_api_model,
            "messages": [{"role": "user", "content": build_student_user_prompt(text)}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens or self.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        try:
            response = self._api_client.post(
                f"{self.api_base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise ProductionExtractionError(f"Student API request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            raw = str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProductionExtractionError(
                f"Student API returned an invalid response payload: {exc}"
            ) from exc
        usage = data.get("usage") if isinstance(data, dict) else None
        generated_tokens = (
            int(usage.get("completion_tokens", 0))
            if isinstance(usage, dict)
            else max(1, len(raw) // 4)
        )
        if not validate:
            return {}, generated_tokens, latency_ms
        try:
            parsed = parse_student_json_output(raw)
            return self._normalize_target(parsed), generated_tokens, latency_ms
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProductionExtractionError(
                f"Student model returned invalid compact JSON: {exc}"
            ) from exc

    def generate(
        self,
        text: str,
        *,
        max_new_tokens: int | None = None,
        validate: bool = True,
    ) -> tuple[dict[str, Any], int, float]:
        if self.api_base_url:
            return self._generate_via_api(
                text, max_new_tokens=max_new_tokens, validate=validate
            )
        prompt = apply_student_chat_template(
            self.tokenizer,
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
        limit = max_new_tokens or self.max_new_tokens
        with self.torch.inference_mode():
            try:
                output_ids = self._fast_greedy_decode(
                    encoded["input_ids"],
                    encoded.get("attention_mask"),
                    limit=limit,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Fast greedy student decode failed (%s); falling back to model.generate.",
                    exc,
                )
                self._cuda_graph = None
                self._static_cache = None
                if self.compiled:
                    self.model.forward = self._eager_forward
                    self.compiled = False
                    if hasattr(self.torch, "_dynamo"):
                        self.torch._dynamo.reset()
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=limit,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids,
                )
                output_ids = generated[0, prompt_tokens:]
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - started) * 1000
        generated_tokens = int(output_ids.shape[-1])
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if not validate:
            return {}, generated_tokens, latency_ms
        try:
            parsed = parse_student_json_output(raw)
            return self._normalize_target(parsed), generated_tokens, latency_ms
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProductionExtractionError(
                f"Student model returned invalid compact JSON: {exc}"
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
        self._http_client: httpx.Client | None = None
        self._models: list[ProductionModelStatus] = []

    def initialize(self) -> None:
        if not self.settings.production_extraction_enabled or self._ready:
            return
        self._loading = True
        self._error = None
        try:
            if self.settings.production_split_runtime:
                self._student = _StudentRuntime(self.settings)
                self._http_client = httpx.Client(
                    timeout=self.settings.production_ocr_service_timeout_seconds
                )
                self._models = [
                    ProductionModelStatus(
                        name="student_extractor",
                        loaded=True,
                        device=self._student.device,
                        model=str(self._student.model_dir),
                        message="OCR is provided by the private OCR service.",
                    )
                ]
                if self.settings.production_warmup_models:
                    self._student.warmup()
                self._ready = True
                LOGGER.info(
                    "Production student runtime is ready; device=%s ocr_service=%s",
                    self._student.device,
                    self.settings.production_ocr_service_base_url,
                )
                return

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
        image = Image.new("RGB", (640, 640), "white")
        try:
            self._classifier.predict_images([image], top_k=1)
            self._orientation.predict_image(image)
            self._ocr.predict_image(image)
        finally:
            image.close()
        self._student.warmup()

    def readiness(self) -> ProductionReadinessResponse:
        if (
            self.settings.production_extraction_enabled
            and self.settings.production_split_runtime
            and self._ready
        ):
            try:
                remote = self._remote_ocr_readiness()
            except (httpx.HTTPError, RuntimeError, TypeError, ValueError) as exc:
                return ProductionReadinessResponse(
                    enabled=True,
                    ready=False,
                    loading=self._loading,
                    seal_detection_enabled=False,
                    models=self._models,
                    error=f"OCR service is unavailable: {exc}",
                )
            return ProductionReadinessResponse(
                enabled=True,
                ready=remote.ready,
                loading=self._loading or remote.loading,
                seal_detection_enabled=False,
                models=[*remote.models, *self._models],
                error=(
                    remote.error
                    or (None if remote.ready else "Private OCR service is not ready.")
                ),
            )
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
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def _remote_ocr_readiness(self) -> ProductionReadinessResponse:
        if self._http_client is None:
            raise RuntimeError("OCR HTTP client has not been initialized.")
        response = self._http_client.get(
            f"{self.settings.production_ocr_service_base_url}/readiness",
            timeout=min(5.0, self.settings.production_ocr_service_timeout_seconds),
        )
        response.raise_for_status()
        return ProductionReadinessResponse.model_validate(response.json())

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
            if self.settings.production_split_runtime:
                return self._extract_split_locked(
                    pdf_path=pdf_path,
                    filename=filename,
                    sha256=sha256,
                )
            return self._extract_locked(pdf_path=pdf_path, filename=filename, sha256=sha256)
        finally:
            self._lock.release()

    def preview_page_classifications(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionPageClassificationResponse:
        if not self._ready:
            raise ProductionExtractionError(
                self._error or "Production extraction models are not loaded yet."
            )
        if not self.settings.production_split_runtime:
            raise ProductionExtractionError(
                "Page classification preview requires the split OCR runtime."
            )
        if self._http_client is None:
            raise ProductionExtractionError("OCR HTTP client has not been initialized.")

        try:
            with pdf_path.open("rb") as source:
                response = self._http_client.post(
                    f"{self.settings.production_ocr_service_base_url}/page-classifications",
                    files={"file": (filename, source, "application/pdf")},
                )
        except httpx.HTTPError as exc:
            raise ProductionExtractionError(f"OCR service request failed: {exc}") from exc
        if response.status_code == 429:
            raise ProductionExtractionError(
                "The OCR GPU worker is busy. Retry this request shortly."
            )
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except (TypeError, ValueError):
                detail = response.text
            raise ProductionExtractionError(
                f"OCR service returned HTTP {response.status_code}: "
                f"{detail or 'unknown error'}"
            )
        try:
            result = ProductionPageClassificationResponse.model_validate(response.json())
        except (TypeError, ValueError) as exc:
            raise ProductionExtractionError(
                f"OCR service returned an invalid classification response: {exc}"
            ) from exc
        if result.sha256 != sha256:
            raise ProductionExtractionError(
                "OCR service response SHA-256 does not match the uploaded PDF."
            )
        return result

    def _extract_split_locked(
        self,
        *,
        pdf_path: Path,
        filename: str,
        sha256: str,
    ) -> ProductionExtractionResponse:
        if self._http_client is None or self._student is None:
            raise ProductionExtractionError("Split production runtime is not initialized.")
        total_started = time.perf_counter()
        request_id = str(uuid4())
        LOGGER.info(
            "Extraction pipeline started request_id=%s filename=%r sha256=%s",
            request_id,
            filename,
            sha256[:12],
        )
        ocr_request_started = time.perf_counter()
        try:
            with pdf_path.open("rb") as source:
                response = self._http_client.post(
                    f"{self.settings.production_ocr_service_base_url}/reconstruct",
                    files={"file": (filename, source, "application/pdf")},
                )
        except httpx.HTTPError as exc:
            LOGGER.exception(
                "Extraction OCR request failed request_id=%s filename=%r",
                request_id,
                filename,
            )
            raise ProductionExtractionError(f"OCR service request failed: {exc}") from exc
        ocr_round_trip_ms = (time.perf_counter() - ocr_request_started) * 1000
        if response.status_code == 429:
            raise ProductionExtractionError(
                "The OCR GPU worker is busy. Retry this request shortly."
            )
        if response.is_error:
            try:
                detail = response.json().get("detail")
            except (TypeError, ValueError):
                detail = response.text
            LOGGER.error(
                "Extraction OCR response failed request_id=%s filename=%r status=%d "
                "round_trip_ms=%.1f detail=%s",
                request_id,
                filename,
                response.status_code,
                ocr_round_trip_ms,
                detail,
            )
            raise ProductionExtractionError(
                f"OCR service returned HTTP {response.status_code}: "
                f"{detail or 'unknown error'}"
            )
        try:
            ocr_result = ProductionOcrResponse.model_validate(response.json())
        except (TypeError, ValueError) as exc:
            raise ProductionExtractionError(
                f"OCR service returned an invalid response: {exc}"
            ) from exc
        if ocr_result.sha256 != sha256:
            raise ProductionExtractionError(
                "OCR service response SHA-256 does not match the uploaded PDF."
            )

        LOGGER.info(
            "Extraction OCR completed request_id=%s filename=%r pages=%d chars=%d "
            "round_trip_ms=%.1f internal_total_ms=%.1f",
            request_id,
            filename,
            ocr_result.page_count,
            len(ocr_result.reconstructed_text),
            ocr_round_trip_ms,
            ocr_result.timings.total_ms,
        )
        compact_output, generated_tokens, extraction_ms = self._student.generate(
            ocr_result.reconstructed_text
        )
        LOGGER.info(
            "Extraction student completed request_id=%s filename=%r generated_tokens=%d "
            "elapsed_ms=%.1f",
            request_id,
            filename,
            generated_tokens,
            extraction_ms,
        )
        extraction = student_target_to_compact_document(
            compact_output,
            case_id=request_id,
            document_id=request_id,
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        LOGGER.info(
            "Extraction pipeline completed request_id=%s filename=%r applicants=%d "
            "total_ms=%.1f",
            request_id,
            filename,
            len(extraction.applicants),
            total_ms,
        )
        return ProductionExtractionResponse(
            request_id=request_id,
            filename=filename,
            sha256=sha256,
            page_count=ocr_result.page_count,
            extraction=extraction,
            compact_model_output=compact_output,
            reconstructed_text=(
                ocr_result.reconstructed_text
                if self.settings.production_include_reconstructed_text
                else None
            ),
            pages=ocr_result.pages,
            timings=ProductionTiming(
                render_ms=ocr_result.timings.render_ms,
                orientation_ms=ocr_result.timings.orientation_ms,
                classification_ms=ocr_result.timings.classification_ms,
                ocr_ms=ocr_result.timings.ocr_ms,
                reconstruction_ms=ocr_result.timings.reconstruction_ms,
                extraction_ms=round(extraction_ms, 3),
                total_ms=round(total_ms, 3),
            ),
            warnings=ocr_result.warnings,
        )

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

        prepared_pages: list[_PreparedPage] = []
        try:
            orientation_started = time.perf_counter()
            prepared_pages = self._orient_pages(rendered)
            orientation_ms = (time.perf_counter() - orientation_started) * 1000

            classification_started = time.perf_counter()
            predictions = self._classifier.predict_images(
                [page.image for page in prepared_pages], top_k=1
            )
            classification_ms = (time.perf_counter() - classification_started) * 1000

            ocr_started = time.perf_counter()
            ocr_pages: list[_OcrPage] = []
            target_types = set(self.settings.production_ocr_page_type_list)
            has_target_page = any(
                prediction.label.value in target_types for prediction in predictions
            )
            fallback_all_pages = bool(prepared_pages) and not has_target_page
            if fallback_all_pages:
                fallback_msg = (
                    "No page matched configured target page types "
                    f"({', '.join(self.settings.production_ocr_page_type_list)}); "
                    f"falling back to OCR on all {len(prepared_pages)} page(s)."
                )
                warnings.append(fallback_msg)
                LOGGER.warning(
                    "OCR stage classification fallback filename=%r pages=%d message=%s",
                    filename,
                    len(prepared_pages),
                    fallback_msg,
                )
            page_results: list[ProductionPageResult] = []
            for page, prediction in zip(prepared_pages, predictions, strict=True):
                page_type = prediction.label.value
                use_page = fallback_all_pages or (page_type in target_types)
                tokens: list[OcrTokenPrediction] = []
                page_ocr_ms = 0.0
                if use_page:
                    page_ocr_started = time.perf_counter()
                    tokens = self._recognize_page(page)
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
        finally:
            for page in prepared_pages:
                page.close()

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
        self, pages: list[RenderedPage]
    ) -> list[_PreparedPage]:
        from app.models.ocr_preprocessing import (
            load_rgb_bytes,
            orientation_probe,
        )

        assert self._orientation is not None
        output: list[_PreparedPage] = []
        for page in pages:
            source = load_rgb_bytes(page.image_bytes)
            probe, detection = orientation_probe(
                source,
                thumbnail_max_side=self.settings.ocr_content_thumbnail_max_side,
                padding_ratio=self.settings.ocr_content_padding_ratio,
                min_components=self.settings.ocr_content_min_components,
                min_content_area_ratio=self.settings.ocr_content_min_area_ratio,
                max_crop_area_ratio=self.settings.ocr_content_max_crop_area_ratio,
            )
            prediction = self._orientation.predict_image(probe)
            probe.close()
            if prediction.angle:
                rotated = source.rotate(prediction.angle, expand=True, fillcolor="white")
                source.close()
                cached_detection = None
            else:
                rotated = source
                cached_detection = detection
            width, height = rotated.size
            output.append(
                _PreparedPage(
                    page_number=page.page_number,
                    image=rotated,
                    image_bytes=None,
                    width=width,
                    height=height,
                    rotation_angle=prediction.angle,
                    orientation_confidence=prediction.confidence,
                    content_detection=cached_detection,
                )
            )
        return output

    def _recognize_page(self, page: _PreparedPage) -> list[OcrTokenPrediction]:
        from app.models.ocr_preprocessing import (
            ocr_quality_score,
            prepare_ocr_image,
            recommended_upscale,
        )
        from app.models.ocr_region_refinement import refine_ocr_tokens

        assert self._ocr is not None
        prepare_kwargs = {
            "rotation": 0,
            "enable_content_crop": self.settings.ocr_enable_content_crop,
            "thumbnail_max_side": self.settings.ocr_content_thumbnail_max_side,
            "padding_ratio": self.settings.ocr_content_padding_ratio,
            "min_components": self.settings.ocr_content_min_components,
            "min_content_area_ratio": self.settings.ocr_content_min_area_ratio,
            "max_crop_area_ratio": self.settings.ocr_content_max_crop_area_ratio,
            "max_processed_side": self.settings.ocr_upscale_max_processed_side,
            "precomputed_detection": page.content_detection,
        }
        selected = prepare_ocr_image(page.image, scale=1.0, **prepare_kwargs)
        tokens = self._ocr.predict_image(selected.image)
        selected_score = ocr_quality_score(tokens)
        if self.settings.production_ocr_enable_small_text_upscale:
            scale = recommended_upscale(
                tokens,
                target_text_height=self.settings.ocr_upscale_target_text_height,
                max_factor=self.settings.ocr_upscale_max_factor,
            )
            if scale > 1.05:
                upscaled = prepare_ocr_image(page.image, scale=scale, **prepare_kwargs)
                upscaled_tokens = self._ocr.predict_image(upscaled.image)
                if ocr_quality_score(upscaled_tokens) >= selected_score - 0.05:
                    selected.image.close()
                    selected = upscaled
                    tokens = upscaled_tokens
                else:
                    upscaled.image.close()
        if self.settings.production_ocr_enable_refinement and tokens:
            with tempfile.TemporaryDirectory(prefix="production-extract-refine-") as refine_dir:
                refinement = refine_ocr_tokens(
                    selected.image,
                    tokens,
                    temp_dir=Path(refine_dir),
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
        return tokens

