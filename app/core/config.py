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
    ocr_det_model_path: Path | None = None
    ocr_rec_model_path: Path | None = None
    ocr_char_dict_path: Path | None = None
    note_extractor_model_path: Path | None = None
    note_extractor_labels_path: Path | None = None
    table_structure_model_path: Path | None = None

    pdf_dpi: int = Field(default=200, ge=72, le=400)
    max_file_size_mb: int = Field(default=30, ge=1, le=500)
    max_pages: int = Field(default=50, ge=1, le=1000)
    max_render_pixels: int = Field(default=25_000_000, ge=1_000_000)
    low_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    mock_page_types: str = (
        "DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST,VISA_APPLICATION_FORM,"
        "TEMPORARY_RESIDENCE_EXTENSION_FORM,PASSPORT_FRONT,PASSPORT_BACK,VISA_STICKER,"
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
