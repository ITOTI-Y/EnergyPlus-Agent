from pathlib import Path
from typing import Final

from src.validator import BaseSchema

MAX_RETRIES: Final[int] = 0

DEFAULT_OUTPUT_DIR: Final[Path] = Path("output")

IDD_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "dependencies"
    / "Energy+.idd"
)


_SCHEMA_INITIALIZED = False


def ensure_schema_initialized() -> None:
    """Load the EnergyPlus IDD into BaseSchema once per process."""
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    if not IDD_PATH.exists():
        raise FileNotFoundError(
            f"Energy+.idd not found at {IDD_PATH}. "
            "Ensure data/dependencies/Energy+.idd exists in the project root."
        )
    BaseSchema.set_idf(IDD_PATH)
    _SCHEMA_INITIALIZED = True
