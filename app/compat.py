from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal stdlib StrEnum compatibility for Python versions before 3.11."""

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
