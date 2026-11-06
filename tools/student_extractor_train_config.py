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
            "STUDENT_TRAIN_GPU_DEVICE_IDS must contain comma-separated integers"
        ) from error
    if any(device_id < 0 for device_id in device_ids):
        raise ValueError("STUDENT_TRAIN_GPU_DEVICE_IDS cannot contain negative IDs")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("STUDENT_TRAIN_GPU_DEVICE_IDS cannot contain duplicate IDs")
    return device_ids


class StudentExtractorTrainSettings(BaseSettings):
    """Offline Gemma student-extractor fine-tuning configuration."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    student_train_model_id: str = "google/gemma-3-270m-it"
    student_train_model_dir: Path | None = None
    student_train_dataset_dir: Path = Path("data/training/gemma3_270m_positional_v4")
    student_train_output_dir: Path = Path(
        "output/student_extractor_gemma3_270m_positional_v4"
    )
    student_train_device: str = "gpu"
    student_train_gpu_device_ids: str = "2,3,4"
    student_train_epochs: int = Field(default=30, ge=1, le=1000)
    student_train_batch_size: int = Field(default=1, ge=1, le=256)
    student_train_eval_batch_size: int = Field(default=2, ge=1, le=256)
    student_train_gradient_accumulation_steps: int = Field(
        default=8, ge=1, le=1024
    )
    student_train_learning_rate: float = Field(default=0.0002, gt=0.0, le=1.0)
    student_train_weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)
    student_train_warmup_ratio: float = Field(default=0.10, ge=0.0, lt=1.0)
    student_train_max_sequence_length: int = Field(default=4096, ge=256, le=32768)
    student_train_num_workers: int = Field(default=4, ge=0, le=64)
    student_train_precision: str = "bf16"
    student_train_lora_rank: int = Field(default=16, ge=1, le=1024)
    student_train_lora_alpha: int = Field(default=32, ge=1, le=4096)
    student_train_lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    student_train_lora_target_modules: str = (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    student_train_early_stopping_patience: int = Field(default=5, ge=0, le=100)
    student_train_save_total_limit: int = Field(default=3, ge=1, le=100)
    student_train_seed: int = Field(default=42, ge=0)
    student_train_gradient_checkpointing: bool = True
    student_train_fail_on_truncation: bool = True
    student_train_merge_after_train: bool = True
    student_train_overwrite_output: bool = False
    student_train_resume_from_checkpoint: Path | None = None
    student_train_offline: bool = False

    @field_validator("student_train_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gpu", "cpu"}:
            raise ValueError("STUDENT_TRAIN_DEVICE must be gpu or cpu")
        return normalized

    @field_validator("student_train_precision")
    @classmethod
    def validate_precision(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"bf16", "fp16", "fp32"}:
            raise ValueError(
                "STUDENT_TRAIN_PRECISION must be bf16, fp16, or fp32"
            )
        return normalized

    @field_validator("student_train_gpu_device_ids")
    @classmethod
    def validate_gpu_device_ids(cls, value: str) -> str:
        return ",".join(str(item) for item in parse_gpu_device_ids(value))

    @field_validator(
        "student_train_model_dir",
        "student_train_resume_from_checkpoint",
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
        return self._resolve(self.student_train_dataset_dir)

    @property
    def output_path(self) -> Path:
        return self._resolve(self.student_train_output_dir)

    @property
    def model_path(self) -> Path | None:
        if self.student_train_model_dir is None:
            return None
        return self._resolve(self.student_train_model_dir)

    @property
    def resume_checkpoint_path(self) -> Path | None:
        if self.student_train_resume_from_checkpoint is None:
            return None
        return self._resolve(self.student_train_resume_from_checkpoint)

    @property
    def gpu_device_ids(self) -> tuple[int, ...]:
        return parse_gpu_device_ids(self.student_train_gpu_device_ids)

    @property
    def lora_target_modules(self) -> tuple[str, ...]:
        modules = tuple(
            item.strip()
            for item in self.student_train_lora_target_modules.split(",")
            if item.strip()
        )
        if not modules:
            raise ValueError("STUDENT_TRAIN_LORA_TARGET_MODULES cannot be empty")
        return modules
