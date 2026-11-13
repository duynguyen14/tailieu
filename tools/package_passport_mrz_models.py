from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the READMRZ YOLO detector to ONNX and package it together "
            "with the exported Paddle CTC recognizer for Linux deployment."
        )
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--recognizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def _export_detector(source: Path, *, image_size: int, opset: int) -> Path:
    if source.suffix.lower() == ".onnx":
        return source
    if source.suffix.lower() != ".pt":
        raise ValueError("Detector must be an Ultralytics .pt or exported .onnx file.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required only on Windows for .pt -> ONNX export. "
            "Install it in the READMRZ environment first."
        ) from exc
    exported = YOLO(str(source)).export(
        format="onnx",
        imgsz=image_size,
        opset=opset,
        simplify=True,
        dynamic=False,
        device="cpu",
    )
    return Path(exported)


def _copy_recognizer(source: Path, destination: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Recognizer directory was not found: {source}")
    model_files = [
        path
        for path in source.iterdir()
        if path.is_file()
        and (
            path.name.startswith("inference")
            or path.suffix.lower() in {".txt", ".yml", ".yaml"}
        )
    ]
    if not any(path.suffix == ".pdiparams" for path in model_files):
        raise FileNotFoundError(
            "Recognizer directory does not contain inference.pdiparams. "
            "Export the Paddle model for inference before packaging."
        )
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source_file in model_files:
        target = destination / source_file.name
        shutil.copy2(source_file, target)
        copied.append(target)
    dictionary = destination / "mrz_dict.txt"
    if not dictionary.is_file():
        candidates = sorted(
            destination.glob("*.txt"),
            key=lambda path: ("dict" not in path.name.casefold(), path.name),
        )
        if not candidates:
            raise FileNotFoundError(
                "Recognizer export does not contain a character dictionary (.txt)."
            )
        shutil.copy2(candidates[0], dictionary)
        copied.append(dictionary)
    inference_config = destination / "inference.yml"
    if not inference_config.is_file():
        candidates = [*destination.glob("*.yml"), *destination.glob("*.yaml")]
        if not candidates:
            raise FileNotFoundError(
                "Recognizer export does not contain inference.yml (or another YAML config)."
            )
        shutil.copy2(candidates[0], inference_config)
        copied.append(inference_config)
    return copied


def main() -> None:
    args = parse_args()
    detector = args.detector.resolve()
    recognizer = args.recognizer_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    exported_detector = _export_detector(
        detector,
        image_size=args.image_size,
        opset=args.opset,
    )
    detector_target = output / "detector.onnx"
    if exported_detector.resolve() != detector_target.resolve():
        shutil.copy2(exported_detector, detector_target)
    copied = _copy_recognizer(recognizer, output / "recognition")

    print(f"Packaged detector: {detector_target}")
    print(f"Packaged recognizer files: {len(copied)}")
    print(f"Deployment directory: {output}")


if __name__ == "__main__":
    main()
