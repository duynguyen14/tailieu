from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import PROJECT_ROOT


class OcrRecognitionDatasetSettings(BaseSettings):
    """Configuration for exporting human-approved OCR recognition crops."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ocr_rec_dataset_export_dir: Path = Path(
        "data/training/ocr_recognition_anotation_2_v1"
    )
    ocr_rec_dataset_source_database: str = "anotation_2"
    ocr_rec_dataset_version: str = "ocr-recognition-anotation2-v1"
    ocr_rec_dataset_split_seed: int = 42
    ocr_rec_dataset_train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    ocr_rec_dataset_val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    ocr_rec_dataset_test_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    ocr_rec_dataset_padding_ratio: float = Field(default=0.05, ge=0.0, le=0.5)
    ocr_rec_dataset_fail_on_invalid: bool = True
    ocr_rec_dataset_deduplicate: bool = True

    @model_validator(mode="after")
    def validate_settings(self) -> "OcrRecognitionDatasetSettings":
        total = (
            self.ocr_rec_dataset_train_ratio
            + self.ocr_rec_dataset_val_ratio
            + self.ocr_rec_dataset_test_ratio
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "OCR_REC_DATASET_TRAIN_RATIO + OCR_REC_DATASET_VAL_RATIO + "
                "OCR_REC_DATASET_TEST_RATIO must be 1."
            )
        if not self.ocr_rec_dataset_source_database.strip():
            raise ValueError("OCR_REC_DATASET_SOURCE_DATABASE cannot be empty.")
        if not self.ocr_rec_dataset_version.strip():
            raise ValueError("OCR_REC_DATASET_VERSION cannot be empty.")
        return self

    @property
    def export_path(self) -> Path:
        if self.ocr_rec_dataset_export_dir.is_absolute():
            return self.ocr_rec_dataset_export_dir.resolve()
        return (PROJECT_ROOT / self.ocr_rec_dataset_export_dir).resolve()
