from app.services.text_reconstruction import (
    PageReconstructionInput,
    ReconstructionElement,
    ReconstructionTable,
    ReconstructionToken,
    group_token_lines,
    reconstruct_page,
    reconstruct_table,
)


def token(
    token_id: str,
    text: str,
    bbox: tuple[int, int, int, int],
    order: int,
) -> ReconstructionToken:
    return ReconstructionToken(id=token_id, text=text, bbox=bbox, order=order)


def test_group_token_lines_restores_reading_order() -> None:
    tokens = [
        token("3", "15/09/2026", (300, 80, 430, 105), 3),
        token("1", "Số:", (40, 40, 90, 65), 1),
        token("2", "1089-26/CDNC-LS", (100, 40, 280, 65), 2),
        token("4", "Ngày gửi:", (180, 80, 290, 105), 4),
    ]

    lines = group_token_lines(tokens)

    assert [[item.text for item in line] for line in lines] == [
        ["Số:", "1089-26/CDNC-LS"],
        ["Ngày gửi:", "15/09/2026"],
    ]


def test_reconstruct_table_uses_reviewed_cells_to_preserve_person_relationships() -> None:
    tokens = [
        token("h1", "Họ và tên", (10, 10, 110, 30), 1),
        token("h2", "Hộ chiếu", (120, 10, 220, 30), 2),
        token("r1c1", "NICHOLAS HOGARTH", (10, 40, 110, 60), 3),
        token("r1c2", "RF0210530", (120, 40, 220, 60), 4),
        token("r2c1", "JENNIFER BLAIR", (10, 70, 110, 90), 5),
        token("r2c2", "RF0209224", (120, 70, 220, 90), 6),
    ]
    cells = tuple(
        ReconstructionElement(
            element_type="CELL",
            row_index=row,
            column_index=column,
            bbox=bbox,
        )
        for row, column, bbox in [
            (0, 0, (0, 0, 115, 35)),
            (0, 1, (115, 0, 230, 35)),
            (1, 0, (0, 35, 115, 65)),
            (1, 1, (115, 35, 230, 65)),
            (2, 0, (0, 65, 115, 100)),
            (2, 1, (115, 65, 230, 100)),
        ]
    )
    table = ReconstructionTable(
        table_index=0,
        bbox=(0, 0, 230, 100),
        elements=(
            ReconstructionElement(
                element_type="HEADER_ROW", row_index=0, bbox=(0, 0, 230, 35)
            ),
            *cells,
        ),
    )

    text, structured, warnings = reconstruct_table(table, tokens)

    assert structured is True
    assert warnings == []
    assert "[HEADER] Họ và tên | Hộ chiếu" in text
    assert "[ROW 1] NICHOLAS HOGARTH | RF0210530" in text
    assert "[ROW 2] JENNIFER BLAIR | RF0209224" in text


def test_reconstruct_page_keeps_plain_text_and_does_not_duplicate_table_tokens() -> None:
    page = PageReconstructionInput(
        page_id="page-1",
        page_number=1,
        page_type="APPLICANT_TABLE",
        page_role="APPLICANT_DATA",
        tokens=(
            token("title", "Đề nghị cấp thị thực cho:", (20, 10, 250, 30), 1),
            token("name", "NGUYEN VAN A", (20, 60, 180, 80), 2),
        ),
        tables=(
            ReconstructionTable(table_index=0, bbox=(0, 40, 300, 100)),
        ),
    )

    result = reconstruct_page(page)

    assert result.text.count("NGUYEN VAN A") == 1
    assert "[TEXT]\nĐề nghị cấp thị thực cho:" in result.text
    assert "[TABLE_TEXT]\nNGUYEN VAN A" in result.text
    assert result.warnings == ("TABLE_STRUCTURE_UNAVAILABLE",)
