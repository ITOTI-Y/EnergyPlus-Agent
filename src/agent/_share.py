import os
from pathlib import Path
from typing import Final

MAX_GLOBAL_RETRIES: Final[int] = 2
"""Intake revisions after validation errors before a human reviews the model."""

DEFAULT_OUTPUT_DIR: Final[Path] = Path("output")

IMAGE_TO_3D_URL: Final[str] = os.getenv("IMAGE_TO_3D_URL", "")
"""TokenHub's HY-3D API (e.g. https://tokenhub.tencentmaas.com/v1/api/3d),
whose mesh of the first image gives the photo-read building's blocks;
empty, the vision model's blocks are kept."""

IMAGE_TO_3D_API_KEY: Final[str] = os.getenv("IMAGE_TO_3D_API_KEY", "")

SEGMENT_URL: Final[str] = os.getenv("SEGMENT_URL", "")
"""SAM 3 server that cuts the photographed building out of its image before
image-to-3D (see src/agent/segmentation.py); empty, the image is cropped to
the building's volumes only."""


AGENT_LANGUAGE: Final[str] = os.getenv("AGENT_LANGUAGE", "English")
"""Language every agent uses for narrative / summary / explanation text.

Applied project-wide by `language_directive()`. Override via the
`AGENT_LANGUAGE` env var (e.g. "English", "日本語"). Empty / "English"
collapses to a no-op.

EnergyPlus identifiers (tool arg values like object names, schedule
names, construction names, IDD enum choices such as 'Outdoors' /
'Weekdays' / 'FullExterior', numeric values) always remain English
ASCII regardless of this setting — they are consumed by the IDF parser,
not by humans.
"""


def language_directive() -> str:
    """Return a system-prompt suffix enforcing AGENT_LANGUAGE.

    Emits an empty string when the configured language is English, so
    existing prompts stay byte-identical in the default case.
    """
    lang = AGENT_LANGUAGE.strip()
    if lang.lower() in {"", "english", "en"}:
        return ""
    return (
        "\n\n=== Language ===\n"
        f"Write all narrative / summary / explanation text in {lang}. "
        f"This includes your final AIMessage, intake `*_specs` strings, "
        f"and any natural-language error reports.\n"
        "HOWEVER, the following MUST remain English ASCII regardless:\n"
        "  - Tool names and argument keys (name, zone_name, variable_name, ...)\n"
        "  - Argument values that are EnergyPlus identifiers (object names,\n"
        "    schedule names, construction names, material names, layer names)\n"
        "  - Enum choices defined by the IDD (e.g. 'Outdoors', 'Weekdays',\n"
        "    'FullExterior', 'MediumRough', 'SunExposed', 'NoSun', 'Yes', 'No')\n"
        "  - Numeric values, paths, file names\n"
        "Those identifiers are read by the EnergyPlus IDF parser, not by\n"
        "humans — never translate or transliterate them.\n"
    )
