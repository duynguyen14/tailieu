from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: paddleocr_entrypoint.py <paddle-script> [args...]")

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"PaddleOCR entry script was not found: {target}")

    # Albumentations exposes optional PyTorch transforms at import time. On
    # Windows, importing Torch after Paddle can load an incompatible cuDNN DLL.
    # PaddleOCR recognition training does not use those PyTorch transforms, so
    # make that optional import raise ImportError and let Albumentations skip it.
    sys.modules.pop("torch", None)
    sys.modules["torch"] = None
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    sys.argv = [str(target), *sys.argv[2:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
