from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch

from tools.document_classifier_config import DocumentClassifierSettings
from tools.document_classifier_model import build_mobilenet_v3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained classifier to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--labels-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DocumentClassifierSettings()
    checkpoint_path = DocumentClassifierSettings.resolve_path(
        args.checkpoint or (config.train_output_dir / "best_model.pt")
    )
    output_path = DocumentClassifierSettings.resolve_path(
        args.output or config.classifier_onnx_output_path
    )
    labels_output_path = DocumentClassifierSettings.resolve_path(
        args.labels_output or config.classifier_labels_output_path
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_to_index = checkpoint["class_to_index"]
    architecture = checkpoint.get("architecture")
    if architecture != "mobilenet_v3_small":
        raise RuntimeError(f"Unsupported checkpoint architecture: {architecture!r}")

    model = build_mobilenet_v3_small(
        num_classes=len(class_to_index),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    image_size = int(checkpoint["image_size"])
    dummy_input = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=config.classifier_onnx_opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    labels = [
        label for label, _ in sorted(class_to_index.items(), key=lambda item: int(item[1]))
    ]
    labels_output_path.write_text(
        json.dumps(
            {
                "architecture": architecture,
                "image_size": image_size,
                "input_name": "images",
                "output_name": "logits",
                "labels": labels,
                "class_to_index": class_to_index,
                "preprocessing": {
                    "color_mode": "RGB",
                    "pad_to_square": True,
                    "pad_value": 255,
                    "resize": [image_size, image_size],
                    "mean": checkpoint["normalization"]["mean"],
                    "std": checkpoint["normalization"]["std"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ONNX model: {output_path}")
    print(f"Labels: {labels_output_path}")


if __name__ == "__main__":
    main()
