from app.core.config import Settings


def test_empty_optional_model_directories_are_none() -> None:
    settings = Settings(
        _env_file=None,
        table_layout_model_dir="",
        table_structure_model_dir="   ",
    )

    assert settings.table_layout_model_dir is None
    assert settings.table_structure_model_dir is None
    assert Settings.resolve_optional_project_path(settings.table_layout_model_dir) is None
    assert Settings.resolve_optional_project_path(settings.table_structure_model_dir) is None
