from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class BarcodeHit:
    page: int
    format: str
    valid: bool
    error: str | None
    text: str
    content_type: str | None
    symbology_identifier: str | None
    raw_base64: str
    raw_hex: str
    payload_kind: str
    position: str | None
    variants: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode barcodes from a Vietnamese visa application form image or PDF. "
            "The tool is read-only and does not write to the database."
        )
    )
    parser.add_argument("input", type=Path, help="Image or PDF to inspect")
    parser.add_argument(
        "--page",
        type=int,
        action="append",
        help="One-based PDF page number. Repeat to scan multiple pages; default: all pages.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF render resolution (default: 300)",
    )
    parser.add_argument(
        "--minimum-image-width",
        type=int,
        default=1800,
        help="Upscale small raster images to at least this width (default: 1800)",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Only scan full-page variants; skip top/bottom barcode crops",
    )
    parser.add_argument(
        "--save-variants",
        type=Path,
        help="Optional directory for saving the exact images sent to the decoder",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human-readable report",
    )
    return parser.parse_args()


def require_runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import pymupdf
        import zxingcpp
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as caught:
        raise RuntimeError(
            "Barcode demo dependencies are missing. Install them with: "
            "python -m pip install -r requirements-barcode.txt"
        ) from caught
    return pymupdf, zxingcpp, Image, ImageEnhance, ImageFilter, ImageOps


def load_pages(
    input_path: Path,
    selected_pages: list[int] | None,
    dpi: int,
    minimum_image_width: int,
    pymupdf: Any,
    image_module: Any,
) -> list[tuple[int, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file was not found: {input_path}")

    if dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if minimum_image_width <= 0:
        raise ValueError("--minimum-image-width must be greater than zero")

    if input_path.suffix.lower() != ".pdf":
        image = image_module.open(input_path).convert("RGB")
        image = upscale_small_image(image, minimum_image_width, image_module)
        return [(1, image)]

    pages: list[tuple[int, Any]] = []
    with pymupdf.open(input_path) as document:
        page_numbers = selected_pages or list(range(1, document.page_count + 1))
        invalid = [page for page in page_numbers if page < 1 or page > document.page_count]
        if invalid:
            raise ValueError(
                f"PDF has {document.page_count} page(s); invalid --page value(s): {invalid}"
            )

        zoom = dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        for page_number in page_numbers:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False, colorspace=pymupdf.csRGB)
            image = image_module.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            pages.append((page_number, image))
    return pages


def upscale_small_image(image: Any, minimum_width: int, image_module: Any) -> Any:
    if image.width >= minimum_width:
        return image
    scale = minimum_width / image.width
    height = max(1, round(image.height * scale))
    return image.resize((minimum_width, height), image_module.Resampling.LANCZOS)


def build_variants(
    image: Any,
    include_crops: bool,
    image_module: Any,
    image_ops: Any,
    image_enhance: Any,
    image_filter: Any,
) -> list[tuple[str, Any]]:
    gray = image_ops.grayscale(image)
    contrast = image_ops.autocontrast(gray, cutoff=1)
    sharpened = image_enhance.Contrast(contrast).enhance(1.35).filter(
        image_filter.UnsharpMask(radius=1.2, percent=160, threshold=3)
    )

    variants: list[tuple[str, Any]] = [
        ("full_rgb", image),
        ("full_contrast", sharpened),
    ]
    if not include_crops:
        return variants

    width, height = image.size
    crop_specs = {
        "top_50": (0, 0, width, round(height * 0.50)),
        "bottom_45": (0, round(height * 0.55), width, height),
        # The official online visa form has a linear tracking barcode in the
        # upper-left and a dense PDF417-like symbol near the lower-left.
        "header_barcode": (
            round(width * 0.04),
            round(height * 0.12),
            round(width * 0.42),
            round(height * 0.23),
        ),
        "footer_barcode": (
            round(width * 0.02),
            round(height * 0.78),
            round(width * 0.52),
            round(height * 0.95),
        ),
    }
    for name, box in crop_specs.items():
        rgb_crop = image.crop(box)
        contrast_crop = sharpened.crop(box)
        if name.endswith("_barcode"):
            border = max(12, round(min(rgb_crop.size) * 0.12))
            rgb_crop = image_ops.expand(rgb_crop, border=border, fill="white")
            contrast_crop = image_ops.expand(contrast_crop, border=border, fill="white")
            target_width = max(1200, rgb_crop.width)
            scale = target_width / rgb_crop.width
            target_height = max(1, round(rgb_crop.height * scale))
            size = (target_width, target_height)
            rgb_crop = rgb_crop.resize(size, image_module.Resampling.LANCZOS)
            contrast_crop = contrast_crop.resize(size, image_module.Resampling.LANCZOS)
        variants.append((f"{name}_rgb", rgb_crop))
        variants.append((f"{name}_contrast", contrast_crop))
    return variants


def maybe_call(value: Any) -> Any:
    return value() if callable(value) else value


def barcode_bytes(barcode: Any) -> bytes:
    raw = maybe_call(getattr(barcode, "bytes", b""))
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    try:
        return bytes(raw)
    except (TypeError, ValueError):
        return b""


def barcode_property(barcode: Any, name: str) -> Any:
    value = getattr(barcode, name, None)
    return maybe_call(value)


def classify_payload(text: str, raw: bytes) -> str:
    stripped = text.strip()
    if stripped.isdigit() and len(stripped) <= 32:
        return "numeric_reference"
    if stripped.startswith(("http://", "https://")):
        return "url"
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            return "json"
    if any(separator in stripped for separator in ("=", "|", "<GS>", "\x1d")):
        return "structured_text_candidate"
    if raw and not stripped:
        return "binary"
    if len(stripped) > 32:
        return "long_text_candidate"
    return "text"


def decode_page(
    page_number: int,
    variants: Iterable[tuple[str, Any]],
    zxingcpp: Any,
    save_directory: Path | None,
) -> list[BarcodeHit]:
    hits: dict[tuple[int, str, bytes, str, str], BarcodeHit] = {}
    if save_directory is not None:
        save_directory.mkdir(parents=True, exist_ok=True)

    text_mode = getattr(getattr(zxingcpp, "TextMode", None), "Escaped", None)
    for variant_name, image in variants:
        if save_directory is not None:
            image.save(save_directory / f"page_{page_number:03d}_{variant_name}.png")

        kwargs: dict[str, Any] = {
            "try_rotate": True,
            "try_downscale": True,
            "try_invert": True,
            # Error candidates are useful in this diagnostic tool. For example,
            # a PDF417 ChecksumError proves that the symbol was located but the
            # scan does not contain enough detail to recover its payload.
            "return_errors": True,
        }
        if text_mode is not None:
            kwargs["text_mode"] = text_mode
        decoded = zxingcpp.read_barcodes(image, **kwargs)

        for barcode in decoded:
            raw = barcode_bytes(barcode)
            text = str(barcode_property(barcode, "text") or "")
            barcode_format = str(barcode_property(barcode, "format") or "UNKNOWN")
            if barcode_format in {"None", "UNKNOWN"}:
                continue
            valid = bool(barcode_property(barcode, "valid"))
            error_value = barcode_property(barcode, "error")
            error = str(error_value) if error_value else None
            key = (page_number, barcode_format, raw, text, error or "")
            if key in hits:
                if variant_name not in hits[key].variants:
                    hits[key].variants.append(variant_name)
                continue

            content_type = barcode_property(barcode, "content_type")
            symbology_identifier = barcode_property(barcode, "symbology_identifier")
            position = barcode_property(barcode, "position")
            hits[key] = BarcodeHit(
                page=page_number,
                format=barcode_format,
                valid=valid,
                error=error,
                text=text,
                content_type=str(content_type) if content_type is not None else None,
                symbology_identifier=(
                    str(symbology_identifier) if symbology_identifier is not None else None
                ),
                raw_base64=base64.b64encode(raw).decode("ascii"),
                raw_hex=raw.hex(),
                payload_kind=classify_payload(text, raw),
                position=str(position) if position is not None else None,
                variants=[variant_name],
            )
    return list(hits.values())


def print_report(input_path: Path, scanned_pages: list[int], hits: list[BarcodeHit]) -> None:
    print(f"Input: {input_path.resolve()}")
    print(f"Scanned pages: {', '.join(map(str, scanned_pages))}")
    valid_count = sum(hit.valid for hit in hits)
    print(f"Valid barcodes: {valid_count}")
    print(f"Decode-error candidates: {len(hits) - valid_count}")
    if not hits:
        print(
            "No barcode was decoded. Try a 300-400 DPI PDF render, a sharper source image, "
            "or inspect --save-variants output."
        )
        return

    for index, hit in enumerate(hits, start=1):
        print(
            f"\n[{index}] page={hit.page} format={hit.format} "
            f"valid={'yes' if hit.valid else 'no'}"
        )
        print(f"    error: {hit.error}")
        print(f"    payload_kind: {hit.payload_kind}")
        print(f"    content_type: {hit.content_type}")
        print(f"    symbology: {hit.symbology_identifier}")
        print(f"    text: {hit.text!r}")
        print(f"    raw_base64: {hit.raw_base64}")
        print(f"    position: {hit.position}")
        print(f"    decoded_from: {', '.join(hit.variants)}")


def run(args: argparse.Namespace) -> list[BarcodeHit]:
    pymupdf, zxingcpp, image_module, image_enhance, image_filter, image_ops = (
        require_runtime()
    )
    pages = load_pages(
        args.input,
        args.page,
        args.dpi,
        args.minimum_image_width,
        pymupdf,
        image_module,
    )
    hits: list[BarcodeHit] = []
    for page_number, image in pages:
        variants = build_variants(
            image,
            include_crops=not args.no_crops,
            image_module=image_module,
            image_ops=image_ops,
            image_enhance=image_enhance,
            image_filter=image_filter,
        )
        page_save_directory = (
            args.save_variants / f"page_{page_number:03d}"
            if args.save_variants is not None
            else None
        )
        hits.extend(
            decode_page(
                page_number,
                variants,
                zxingcpp,
                page_save_directory,
            )
        )

    if args.json:
        print(
            json.dumps(
                {
                    "input": str(args.input.resolve()),
                    "scanned_pages": [page_number for page_number, _ in pages],
                    "barcodes": [asdict(hit) for hit in hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_report(args.input, [page_number for page_number, _ in pages], hits)
    return hits


def main() -> None:
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as caught:
        raise SystemExit(f"Visa application barcode test could not start: {caught}") from caught


if __name__ == "__main__":
    main()
