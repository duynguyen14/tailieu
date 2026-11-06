from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import PROJECT_ROOT


def parse_gpu_device_ids(value: str) -> tuple[int, ...]:
    normalized = value.strip()
    if not normalized:
        return ()
    try:
        device_ids = tuple(int(part.strip()) for part in normalized.split(","))
    except ValueError as error:
        raise ValueError(
            "KIE_TRAIN_GPU_DEVICE_IDS must contain comma-separated integers"
        ) from error
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("KIE_TRAIN_GPU_DEVICE_IDS cannot contain negative IDs")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("KIE_TRAIN_GPU_DEVICE_IDS cannot contain duplicate IDs")
    return device_ids


class KieSerTrainSettings(BaseSettings):
    """LayoutLMv3 SER fine-tuning configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    kie_train_model_id: str = "microsoft/layoutlmv3-base"
    kie_train_model_dir: Path | None = None
    kie_train_dataset_dir: Path = Path(
        "data/training/kie_ser_anotation_2_poc_v1"
    )
    kie_train_output_dir: Path = Path(
        "output/kie_ser_layoutlmv3_anotation_2_poc_v1"
    )
    kie_train_device: str = "gpu"
    kie_train_gpu_device_ids: str = "2,3,4"
    kie_train_epochs: int = Field(default=50, ge=1, le=1000)
    kie_train_batch_size: int = Field(default=1, ge=1, le=128)
    kie_train_eval_batch_size: int = Field(default=2, ge=1, le=128)
    kie_train_gradient_accumulation_steps: int = Field(default=1, ge=1, le=1024)
    kie_train_learning_rate: float = Field(default=0.00005, gt=0.0, le=1.0)
    kie_train_weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    kie_train_warmup_ratio: float = Field(default=0.10, ge=0.0, lt=1.0)
    kie_train_max_sequence_length: int = Field(default=512, ge=32, le=512)
    kie_train_num_workers: int = Field(default=2, ge=0, le=64)
    kie_train_precision: str = "fp16"
    kie_train_early_stopping_patience: int = Field(default=7, ge=0, le=100)
    kie_train_save_total_limit: int = Field(default=3, ge=1, le=100)
    kie_train_seed: int = Field(default=42, ge=0)
    kie_train_fail_on_truncation: bool = True
    kie_train_fail_on_missing_train_label: bool = True
    kie_train_overwrite_output: bool = False
    kie_train_resume_from_checkpoint: Path | None = None
    kie_train_offline: bool = False

    @field_validator("kie_train_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gpu", "cpu"}:
            raise ValueError("KIE_TRAIN_DEVICE must be gpu or cpu")
        return normalized

    @field_validator("kie_train_precision")
    @classmethod
    def validate_precision(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"bf16", "fp16", "fp32"}:
            raise ValueError("KIE_TRAIN_PRECISION must be bf16, fp16, or fp32")
        return normalized

    @field_validator("kie_train_gpu_device_ids")
    @classmethod
    def validate_gpu_device_ids(cls, value: str) -> str:
        return ",".join(str(item) for item in parse_gpu_device_ids(value))

    @field_validator(
        "kie_train_model_dir",
        "kie_train_resume_from_checkpoint",
        mode="before",
    )
    @classmethod
    def empty_optional_path_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def dataset_path(self) -> Path:
        return self._resolve(self.kie_train_dataset_dir)

    @property
    def output_path(self) -> Path:
        return self._resolve(self.kie_train_output_dir)

    @property
    def model_path(self) -> Path | None:
        if self.kie_train_model_dir is None:
            return None
        return self._resolve(self.kie_train_model_dir)

    @property
    def resume_checkpoint_path(self) -> Path | None:
        if self.kie_train_resume_from_checkpoint is None:
            return None
        return self._resolve(self.kie_train_resume_from_checkpoint)

    @property
    def gpu_device_ids(self) -> tuple[int, ...]:
        return parse_gpu_device_ids(self.kie_train_gpu_device_ids)
