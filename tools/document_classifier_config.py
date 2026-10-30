from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import PROJECT_ROOT


class DocumentClassifierSettings(BaseSettings):
    """Configuration shared by dataset export, training and ONNX export tools."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dataset_export_dir: Path = Path("data/training/document_classifier")
    dataset_copy_mode: Literal["copy", "hardlink"] = "copy"
    dataset_split_seed: int = 42
    dataset_train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    dataset_val_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    dataset_test_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    dataset_fail_on_invalid: bool = True

    train_output_dir: Path = Path("output/document_classifier")
    train_model_name: Literal["mobilenet_v3_small"] = "mobilenet_v3_small"
    train_image_size: int = Field(default=224, ge=96, le=1024)
    train_batch_size: int = Field(default=32, ge=1)
    train_epochs: int = Field(default=30, ge=1)
    train_learning_rate: float = Field(default=3e-4, gt=0.0)
    train_weight_decay: float = Field(default=1e-4, ge=0.0)
    train_num_workers: int = Field(default=4, ge=0)
    train_device: Literal["auto", "cpu", "cuda"] = "auto"
    train_gpu_device_id: int = Field(default=0, ge=0)
    train_mixed_precision: bool = True
    train_pretrained: bool = True
    train_pretrained_weights_path: Path | None = None
    train_early_stopping_patience: int = Field(default=7, ge=0)
    train_seed: int = 42
    train_deterministic: bool = False

    classifier_onnx_output_path: Path = Path("models/page_classifier/model.onnx")
    classifier_labels_output_path: Path = Path("models/page_classifier/labels.json")
    classifier_onnx_opset: int = Field(default=17, ge=13, le=21)

    @model_validator(mode="after")
    def validate_split_ratios(self) -> "DocumentClassifierSettings":
        total = self.dataset_train_ratio + self.dataset_val_ratio + self.dataset_test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("DATASET_TRAIN_RATIO + DATASET_VAL_RATIO + DATASET_TEST_RATIO must be 1.")
        return self

    @staticmethod
    def resolve_path(path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (PROJECT_ROOT / path).resolve()

    @property
    def dataset_export_path(self) -> Path:
        return self.resolve_path(self.dataset_export_dir)

    @property
    def train_output_path(self) -> Path:
        return self.resolve_path(self.train_output_dir)

    @property
    def pretrained_weights_path(self) -> Path | None:
        if self.train_pretrained_weights_path is None:
            return None
        return self.resolve_path(self.train_pretrained_weights_path)

    @property
    def onnx_output_path(self) -> Path:
        return self.resolve_path(self.classifier_onnx_output_path)

    @property
    def labels_output_path(self) -> Path:
        return self.resolve_path(self.classifier_labels_output_path)

