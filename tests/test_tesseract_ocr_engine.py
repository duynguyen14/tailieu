from pathlib import Path

from PIL import Image, ImageDraw

from app.models.tesseract_ocr_engine import (
    TesseractOcrEngine,
    detect_table_cells,
    parse_tesseract_data,
)


def test_parse_tesseract_data_groups_words_into_lines() -> None:
    data = {
        "text": ["Bản", "dịch", "850-2026", ""],
        "conf": ["91.0", "93.0", "98.0", "-1"],
        "left": [10, 55, 10, 0],
        "top": [20, 20, 60, 0],
        "width": [35, 45, 90, 0],
        "height": [18, 18, 20, 0],
        "page_num": [1, 1, 1, 1],
        "block_num": [1, 1, 1, 0],
        "par_num": [1, 1, 1, 0],
        "line_num": [1, 1, 2, 0],
    }

    tokens = parse_tesseract_data(data)

    assert [token.text for token in tokens] == ["Bản dịch", "850-2026"]
    assert tokens[0].bbox == (10, 20, 100, 38)
    assert tokens[1].bbox == (10, 60, 100, 80)
    assert tokens[0].token_id == "t_0001"
    assert 0.91 <= tokens[0].confidence <= 0.93


def test_detect_table_cells_finds_simple_grid() -> None:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    for x in (100, 300, 500):
        draw.line((x, 80, x, 320), fill="black", width=4)
    for y in (80, 160, 240, 320):
        draw.line((100, y, 500, y), fill="black", width=4)

    cells = detect_table_cells(image, max_cells=20)

    image.close()
    assert len(cells) >= 6
    assert all(x2 > x1 and y2 > y1 for x1, y1, x2, y2 in cells)


def test_tesseract_config_does_not_embed_literal_quotes() -> None:
    class SettingsStub:
        tesseract_oem = 1

    engine = object.__new__(TesseractOcrEngine)
    engine._settings = SettingsStub()  # type: ignore[attr-defined]

    data_dir = Path("C:/TesseractData/tessdata")
    config = engine._config(data_dir=data_dir, psm=3)

    assert config.startswith(f"--tessdata-dir {data_dir} ")
    assert '"' not in config
