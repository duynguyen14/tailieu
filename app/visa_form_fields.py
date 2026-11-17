from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateField:
    code: str
    label: str
    normalized_bbox: tuple[int, int, int, int]


# Coordinates use the 0..1000 annotation space and describe value areas.
VISA_FORM_TEMPLATE_FIELDS: tuple[TemplateField, ...] = (
    TemplateField("full_name", "Họ và tên", (145, 176, 390, 200)),
    TemplateField("date_of_birth", "Ngày sinh", (160, 211, 350, 235)),
    TemplateField("place_of_birth", "Nơi sinh", (160, 247, 390, 271)),
    TemplateField("gender", "Giới tính", (135, 283, 300, 307)),
    TemplateField("nationality", "Quốc tịch", (160, 318, 350, 343)),
    TemplateField("contact_address", "Địa chỉ liên hệ", (75, 375, 440, 420)),
    TemplateField("contact_phone", "Điện thoại liên hệ", (145, 420, 355, 445)),
    TemplateField("contact_email", "Email liên hệ", (95, 453, 410, 480)),
    TemplateField("sponsor_name", "Tên người/cơ quan mời", (125, 500, 420, 532)),
    TemplateField("sponsor_phone", "Điện thoại người/cơ quan mời", (135, 538, 365, 566)),
    TemplateField("passport_number", "Số hộ chiếu", (625, 211, 870, 236)),
    TemplateField("passport_expiration", "Hạn hộ chiếu", (635, 247, 875, 272)),
    TemplateField("number_of_entries", "Số lần nhập cảnh", (635, 283, 875, 308)),
    TemplateField("entry_date", "Ngày dự kiến nhập cảnh", (650, 319, 885, 344)),
    TemplateField("exit_date", "Ngày dự kiến xuất cảnh", (650, 354, 885, 379)),
    TemplateField("submission_place", "Nơi nộp hồ sơ", (515, 405, 890, 440)),
    TemplateField("collection_place", "Nơi nhận thị thực", (515, 474, 890, 508)),
    TemplateField("purpose_of_entry", "Mục đích nhập cảnh", (625, 520, 975, 565)),
    TemplateField("application_date", "Ngày làm đơn", (650, 690, 900, 725)),
)

FIELD_CODES = tuple(item.code for item in VISA_FORM_TEMPLATE_FIELDS)
