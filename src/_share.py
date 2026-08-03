from __future__ import annotations

from typing import ClassVar, Final


class _UnsetType:
    """Sentinel indicating that a value was not provided."""

    _instance: ClassVar[_UnsetType | None] = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: Final = _UnsetType()
