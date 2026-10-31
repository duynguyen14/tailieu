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

    # GPU-only OCR used to create grounded LLM inputs. PaddleOCR returns text,
    # confidence and source boxes; the LLM must cite token IDs instead of
    # generating coordinates itself.
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
    ocr_preprocess_version: str = "orientation-v1"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
