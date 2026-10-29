from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath


class StoragePathError(ValueError):
    pass


class AnnotationStorage:
    """Resolve project-relative storage keys without exposing absolute paths to the DB."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize(relative_path: str | PurePosixPath) -> str:
        raw = str(relative_path).replace("\\", "/")
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise StoragePathError("Storage path must be relative and cannot contain '..'.")
        return path.as_posix()

    def resolve(self, relative_path: str | PurePosixPath) -> Path:
        normalized = self.normalize(relative_path)
        absolute = (self.root / Path(normalized)).resolve()
        if not absolute.is_relative_to(self.root):
            raise StoragePathError("Resolved storage path escaped the configured root.")
        return absolute

    def write_bytes(self, relative_path: str, content: bytes) -> str:
        normalized = self.normalize(relative_path)
        destination = self.resolve(normalized)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return normalized

    def read_bytes(self, relative_path: str) -> bytes:
        return self.resolve(relative_path).read_bytes()

    def remove_file(self, relative_path: str) -> None:
        path = self.resolve(relative_path)
        if path.exists() and path.is_file():
            path.unlink()

    def remove_tree(self, relative_path: str) -> None:
        path = self.resolve(relative_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
