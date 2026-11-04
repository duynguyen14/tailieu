from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from os.path import abspath
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelBackend(StrEnum):
    MOCK = "mock"
    ONNX = "onnx"
    PADDLE = "paddle"
    TENSORRT = "tensorrt"


class DevicePreference(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class OcrEngineKind(StrEnum):
    PADDLE = "paddle"
    TESSERACT = "tesseract"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Document Extraction API"
    app_env: str = "development"
    api_prefix: str = "/api/v1"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_reload: bool = False
    log_level: str = "INFO"

    # DATABASE_URL is intentionally optional and exists for isolated tests or advanced overrides.
    # Normal deployments build a safe SQL Server URL from the DB_* environment variables below.
    database_url: str | None = None
    db_server: str = "localhost"
    db_port: int | None = Field(default=1433, ge=1, le=65535)
    db_name: str = "document_annotation"
    db_user: str | None = "sa"
    db_password: SecretStr | None = None
    db_driver: str = "ODBC Driver 18 for SQL Server"
    db_encrypt: bool = True
    db_trust_server_certificate: bool = True
    db_trusted_connection: bool = False
    auto_create_db: bool = False
    annotation_storage_dir: Path = Path("data/annotation_storage")
    bulk_import_root: Path = Path("data/bulk_import_inbox")
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    model_backend: ModelBackend = ModelBackend.MOCK
    device: DevicePreference = DevicePreference.CPU
    gpu_device_id: int = Field(default=0, ge=0)
    allow_cpu_fallback: bool = True

    enable_page_classifier: bool = True
    enable_ocr: bool = True
    enable_note_extractor: bool = True
    enable_table_structure: bool = True

    page_classifier_model_path: Path | None = None
    page_classifier_labels_path: Path | None = None
    page_classifier_model_version: str = "mobilenet-v3-phase1"
    page_classifier_batch_size: int = Field(default=64, ge=1, le=1024)
    ocr_det_model_path: Path | None = None
    ocr_rec_model_path: Path | None = None
    ocr_char_dict_path: Path | None = None
    note_extractor_model_path: Path | None = None
    note_extractor_labels_path: Path | None = None
    table_structure_model_path: Path | None = None

    # Offline table annotation baseline. These models are loaded only by the
    # tools.infer_table_structure command, never during API startup.
    table_device: DevicePreference = DevicePreference.CPU
    table_gpu_device_id: int = Field(default=0, ge=0)
    table_allow_cpu_fallback: bool = True
    table_layout_model_name: str = "PP-DocLayout-L"
    table_layout_model_dir: Path | None = None
    table_structure_model_name: str = "SLANet_plus"
    table_structure_model_dir: Path | None = None
    table_model_version: str = "paddle-table-baseline-v1"
    table_layout_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    table_cpu_threads: int = Field(default=8, ge=1, le=128)
    table_enable_hpi: bool = False
    table_assume_first_row_header: bool = True

    # Deterministic OCR token -> structured plain-text reconstruction.
    text_reconstruction_version: str = "text-reconstruction-v2"
    text_reconstruction_line_tolerance: float = Field(default=0.55, ge=0.1, le=1.5)

    # GPU-only OCR used to create grounded LLM inputs. PaddleOCR returns text,
    # confidence and source boxes; the LLM must cite token IDs instead of
    # generating coordinates itself.
    ocr_engine: OcrEngineKind = OcrEngineKind.PADDLE
    ocr_runtime_device: DevicePreference = DevicePreference.CUDA
    ocr_runtime_gpu_device_id: int = Field(default=0, ge=0)
    ocr_runtime_allow_cpu_fallback: bool = False
    ocr_detection_model_name: str = "PP-OCRv6_medium_det"
    ocr_recognition_model_name: str = "PP-OCRv6_medium_rec"
    ocr_detection_model_dir: Path | None = None
    ocr_recognition_model_dir: Path | None = None
    ocr_model_version: str = "pp-ocrv6-medium-v1"
    ocr_enable_hpi: bool = False
    ocr_cpu_threads: int = Field(default=8, ge=1, le=128)
    ocr_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    ocr_target_page_types: str = "DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST"
    ocr_enable_exclusion_detection: bool = True
    ocr_layout_model_name: str = "PP-DocLayout-L"
    ocr_layout_model_dir: Path | None = None
    ocr_layout_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    ocr_exclusion_region_types: str = "seal"
    ocr_auto_ignore_overlap: float = Field(default=0.70, ge=0.0, le=1.0)
    ocr_enable_orientation: bool = True
    ocr_orientation_model_name: str = "PP-LCNet_x1_0_doc_ori"
    ocr_orientation_model_dir: Path | None = None
    ocr_orientation_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    ocr_orientation_fallback_all_angles: bool = True
    ocr_enable_content_crop: bool = True
    ocr_content_thumbnail_max_side: int = Field(default=1400, ge=320, le=4000)
    ocr_content_padding_ratio: float = Field(default=0.05, ge=0.0, le=0.25)
    ocr_content_min_components: int = Field(default=8, ge=1, le=1000)
    ocr_content_min_area_ratio: float = Field(default=0.0002, ge=0.0, le=0.10)
    ocr_content_max_crop_area_ratio: float = Field(default=0.95, ge=0.10, le=1.0)
    ocr_enable_small_text_upscale: bool = True
    ocr_upscale_target_text_height: int = Field(default=30, ge=12, le=96)
    ocr_upscale_max_factor: float = Field(default=2.5, ge=1.0, le=4.0)
    ocr_upscale_max_processed_side: int = Field(default=5000, ge=1000, le=12000)
    ocr_preprocess_version: str = "orientation-content-v2"
    # Conservative second pass for Vietnamese/small/low-confidence regions.
    # It reuses the warm Paddle pipeline and never changes source coordinates.
    ocr_enable_selective_refinement: bool = True
    ocr_refinement_version: str = "selective-multiscale-v1"
    ocr_refinement_scales: str = "2.0,3.0"
    ocr_refinement_confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    ocr_refinement_candidate_min_confidence: float = Field(
        default=0.70, ge=0.0, le=1.0
    )
    ocr_refinement_agreement_similarity: float = Field(default=0.96, ge=0.0, le=1.0)
    ocr_refinement_min_confidence_gain: float = Field(default=0.05, ge=0.0, le=1.0)
    ocr_refinement_max_regions: int = Field(default=32, ge=0, le=256)
    ocr_refinement_max_seconds: float = Field(default=15.0, ge=0.1, le=300.0)
    ocr_refinement_padding_ratio: float = Field(default=0.12, ge=0.0, le=0.75)
    ocr_refinement_max_crop_side: int = Field(default=2400, ge=320, le=8000)
    ocr_refinement_protect_digits: bool = True

    # Optional CPU Tesseract backend. Python owns orchestration and SQL writes;
    # pytesseract only invokes the local Tesseract 5 executable.
    tesseract_cmd: Path = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
    tesseract_model_version: str = "tesseract5-fast4.1-vie-best4.1-v1"
    tesseract_fast_data_dir: Path = Path("C:/Program Files/Tesseract-OCR/tessdata")
    tesseract_best_data_dir: Path = Path(
        "C:/Program Files/Tesseract-OCR/tessdata_best"
    )
    tesseract_languages: str = "vie+eng+deu+fra"
    tesseract_oem: int = Field(default=1, ge=0, le=3)
    tesseract_page_segmentation_mode: int = Field(default=3, ge=0, le=13)
    tesseract_refinement_page_segmentation_mode: int = Field(
        default=7, ge=0, le=13
    )
    tesseract_timeout_seconds: float = Field(default=90.0, ge=1.0, le=600.0)
    tesseract_threads: int = Field(default=2, ge=1, le=32)
    tesseract_enable_vie_refinement: bool = True
    tesseract_enable_table_cells: bool = True
    tesseract_table_max_cells: int = Field(default=80, ge=0, le=500)
    tesseract_table_budget_seconds: float = Field(default=20.0, ge=0.1, le=300.0)
    tesseract_use_paddle_aux_models: bool = False
    tesseract_try_all_orientations: bool = True
    paddle_pdx_model_source: str = "BOS"
    paddle_pdx_disable_model_source_check: bool = True

    # Grounded generative extractor. Qwen3-4B with 4-bit QLoRA is the quality /
    # memory balance selected for the RTX 3060 12 GB server.
    llm_base_model_id: str = "Qwen/Qwen3-4B"
    llm_base_model_dir: Path = Path("models/llm/qwen3-4b")
    llm_device: DevicePreference = DevicePreference.AUTO
    llm_gpu_device_id: int = Field(default=0, ge=0)
    llm_load_in_4bit: bool = True
    llm_max_sequence_length: int = Field(default=4096, ge=512, le=32768)
    llm_schema_version: str = "document-case-v1"
    llm_dataset_dir: Path = Path("data/training/grounded_llm")
    llm_train_batch_size: int = Field(default=1, ge=1, le=16)
    llm_gradient_accumulation_steps: int = Field(default=8, ge=1, le=128)
    llm_lora_rank: int = Field(default=16, ge=1, le=256)
    llm_lora_alpha: int = Field(default=32, ge=1, le=512)
    llm_lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)

    pdf_dpi: int = Field(default=200, ge=72, le=400)
    max_file_size_mb: int = Field(default=30, ge=1, le=500)
    max_pages: int = Field(default=50, ge=1, le=1000)
    max_render_pixels: int = Field(default=25_000_000, ge=1_000_000)
    low_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    mock_page_types: str = (
        "DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST,VISA_APPLICATION_FORM,"
        "TEMPORARY_RESIDENCE_EXTENSION_FORM,PASSPORT_FRONT,VISA_STICKER,"
        "VISA_REQUEST_OFFICIAL_LETTER,OTHER"
    )

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper().strip()

    @field_validator(
        "page_classifier_model_path",
        "page_classifier_labels_path",
        "ocr_det_model_path",
        "ocr_rec_model_path",
        "ocr_char_dict_path",
        "note_extractor_model_path",
        "note_extractor_labels_path",
        "table_structure_model_path",
        "table_layout_model_dir",
        "table_structure_model_dir",
        "ocr_detection_model_dir",
        "ocr_recognition_model_dir",
        "ocr_layout_model_dir",
        "ocr_orientation_model_dir",
        mode="before",
    )
    @classmethod
    def empty_optional_path_is_none(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("annotation_storage_dir")
    @classmethod
    def validate_relative_storage_dir(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("ANNOTATION_STORAGE_DIR must be a project-relative path without '..'.")
        normalized = Path(*[part for part in value.parts if part not in {"", "."}])
        if not normalized.parts:
            raise ValueError("ANNOTATION_STORAGE_DIR cannot be empty.")
        return normalized

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def mock_page_type_list(self) -> list[str]:
        return [item.strip().upper() for item in self.mock_page_types.split(",") if item.strip()]

    @property
    def ocr_target_page_type_list(self) -> list[str]:
        return [
            item.strip().upper()
            for item in self.ocr_target_page_types.split(",")
            if item.strip()
        ]

    @property
    def ocr_exclusion_region_type_list(self) -> list[str]:
        return [
            item.strip().upper()
            for item in self.ocr_exclusion_region_types.split(",")
            if item.strip()
        ]

    @property
    def ocr_refinement_scale_list(self) -> list[float]:
        values: list[float] = []
        for raw_value in self.ocr_refinement_scales.split(","):
            raw_value = raw_value.strip()
            if not raw_value:
                continue
            value = float(raw_value)
            if value < 1.0 or value > 6.0:
                raise ValueError("OCR_REFINEMENT_SCALES values must be between 1 and 6.")
            if value not in values:
                values.append(value)
        if self.ocr_enable_selective_refinement and len(values) < 2:
            raise ValueError(
                "OCR_REFINEMENT_SCALES must contain at least two distinct scales."
            )
        return values

    @property
    def tesseract_language_list(self) -> list[str]:
        return [
            item.strip().lower()
            for item in self.tesseract_languages.split("+")
            if item.strip()
        ]

    @property
    def tesseract_cmd_path(self) -> Path:
        return self.resolve_project_path(self.tesseract_cmd)

    @property
    def tesseract_fast_data_path(self) -> Path:
        return self.resolve_project_path(self.tesseract_fast_data_dir)

    @property
    def tesseract_best_data_path(self) -> Path:
        return self.resolve_project_path(self.tesseract_best_data_dir)

    @property
    def llm_base_model_path(self) -> Path:
        path = self.llm_base_model_dir
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def llm_dataset_path(self) -> Path:
        path = self.llm_dataset_dir
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def annotation_storage_path(self) -> Path:
        path = (PROJECT_ROOT / self.annotation_storage_dir).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Annotation storage must stay inside the project root.")
        return path

    @property
    def bulk_import_root_path(self) -> Path:
        if self.bulk_import_root.is_absolute():
            return Path(abspath(self.bulk_import_root))
        return Path(abspath(PROJECT_ROOT / self.bulk_import_root))

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def configured_model_paths(self) -> dict[str, tuple[bool, Path | None]]:
        return {
            "page_classifier": (
                self.enable_page_classifier,
                self.page_classifier_model_path,
            ),
            "ocr_detection": (self.enable_ocr, self.ocr_det_model_path),
            "ocr_recognition": (self.enable_ocr, self.ocr_rec_model_path),
            "note_extractor": (
                self.enable_note_extractor,
                self.note_extractor_model_path,
            ),
            "table_structure": (
                self.enable_table_structure,
                self.table_structure_model_path,
            ),
        }

    @property
    def page_classifier_model_resolved_path(self) -> Path | None:
        return self.resolve_optional_project_path(self.page_classifier_model_path)

    @property
    def page_classifier_labels_resolved_path(self) -> Path | None:
        return self.resolve_optional_project_path(self.page_classifier_labels_path)

    @staticmethod
    def resolve_optional_project_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()

    @staticmethod
    def resolve_project_path(path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
