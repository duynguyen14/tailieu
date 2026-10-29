from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderedPage:
    page_index: int
    page_number: int
    width: int
    height: int
    image_bytes: bytes
    mime_type: str = "image/png"
