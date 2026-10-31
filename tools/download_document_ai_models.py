from __future__ import annotations

import argparse

from app.core.config import DevicePreference, get_settings
from app.models.paddle_ocr_engine import PaddleOcrEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/verify OCR and grounded LLM models without running documents."
    )
    parser.add_argument(
        "--component", choices=("all", "ocr", "llm"), default="all"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.component in {"all", "ocr"}:
        override = DevicePreference(args.device) if args.device else None
        engine = PaddleOcrEngine(settings, device_override=override)
        print(
            f"OCR models ready: {engine.model_name}; device={engine.effective_device}"
        )

    if args.component in {"all", "llm"}:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is missing. Install requirements-llm-gpu.txt first."
            ) from exc
        target = settings.llm_base_model_path
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=settings.llm_base_model_id,
            local_dir=target,
        )
        print(f"LLM model ready: {settings.llm_base_model_id}; path={target}")


if __name__ == "__main__":
    main()

