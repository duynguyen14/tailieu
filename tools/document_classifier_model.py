from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet_v3_small(
    *,
    num_classes: int,
    pretrained: bool,
    pretrained_weights_path: Path | None = None,
) -> nn.Module:
    """Build MobileNetV3-Small and replace its ImageNet classifier head."""

    if pretrained_weights_path is not None:
        model = mobilenet_v3_small(weights=None)
        state = torch.load(pretrained_weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    else:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model

