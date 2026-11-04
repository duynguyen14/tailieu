from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import PROJECT_ROOT


class OcrRecognitionTrainSettings(BaseSettings):
    """Runtime configuration for fine-tuning PaddleOCR recognition."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ocr_train_paddleocr_dir: Path = Path("third_party/PaddleOCR")
    ocr_train_dataset_dir: Path = Path(
        "data/training/ocr_recognition_anotation_2_v1"
    )
    ocr_train_output_dir: Path = Path(
        "output/ocr_recognition_latin_ppocrv5_mobile_v1"
    )
    ocr_train_model_name: str = "latin_PP-OCRv5_mobile_rec"
    ocr_train_config_relative_path: Path = Path(
        "configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml"
    )
    ocr_train_pretrained_url: str = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
        "official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams"
    )
    ocr_train_pretrained_path: Path = Path(
        "models/pretrained/latin_PP-OCRv5_mobile_rec_pretrained.pdparams"
    )
    ocr_train_device: str = "gpu"
    ocr_train_gpu_device_id: int = Field(default=0, ge=0)
    ocr_train_epochs: int = Field(default=50, ge=1, le=1000)
    ocr_train_batch_size: int = Field(default=64, ge=1, le=2048)
    ocr_train_num_workers: int = Field(default=4, ge=0, le=64)
    ocr_train_learning_rate: float = Field(default=0.0001, gt=0.0, le=1.0)
    ocr_train_max_text_length: int = Field(default=160, ge=1, le=2048)
    ocr_train_image_height: int = Field(default=48, ge=16, le=256)
    ocr_train_image_width: int = Field(default=640, ge=64, le=4096)
    ocr_train_eval_batch_step: int = Field(default=500, ge=1)
    ocr_train_save_epoch_step: int = Field(default=5, ge=1)
    ocr_train_use_visualdl: bool = False
    ocr_train_auto_download_pretrained: bool = True
    ocr_train_bootstrap_repo: bool = False
    ocr_train_export_after_train: bool = True
    ocr_train_extend_character_dictionary: bool = True
    ocr_train_fail_on_unknown_characters: bool = True
    ocr_train_checkpoint: Path | None = None

    @field_validator("ocr_train_device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gpu", "cpu"}:
            raise ValueError("OCR_TRAIN_DEVICE must be gpu or cpu")
        return normalized

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def paddleocr_path(self) -> Path:
        return self._resolve(self.ocr_train_paddleocr_dir)

    @property
    def dataset_path(self) -> Path:
        return self._resolve(self.ocr_train_dataset_dir)

    @property
    def output_path(self) -> Path:
        return self._resolve(self.ocr_train_output_dir)

    @property
    def pretrained_path(self) -> Path:
        return self._resolve(self.ocr_train_pretrained_path)

    @property
    def checkpoint_path(self) -> Path | None:
        if self.ocr_train_checkpoint is None:
            return None
        return self._resolve(self.ocr_train_checkpoint)

