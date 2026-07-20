"""Parser for EnergyPlus ``eplusout.err`` severity diagnostics.

EnergyPlus writes one line per diagnostic, tagged like::

    ** Fatal **  GetSurfaceData: Errors discovered, program terminates.
    ** Severe **  checkSubSurfAzTiltNorm: Outward facing angle of subsurface ...
    ** Warning **  ... informational, tolerated ...

The number of spaces between the keyword and the ``**`` markers varies
(e.g. ``** Fatal **`` vs ``** Severe  **``), so we match with a flexible
regex anchored at the line start.

This module is shared between the agent (``simulate_node`` decides whether
to roll back to ``revise`` based on Fatal/Severe presence) and the
robustness benchmark (which records the counts per case).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Anchored regex: optional leading space, "**", flexible whitespace, the
# severity keyword, flexible whitespace, "**". Case-insensitive.
_SEVERITY_RE = re.compile(r"\s*\*\*\s*(Fatal|Severe|Warning)\s*\*\*", re.IGNORECASE)


def extract_errors(err_path: Path | str | None) -> dict[str, Any]:
    """Parse ``eplusout.err`` and return severity counts + raw line texts.

    Returns a dict with:
    - ``fatal`` / ``severe`` / ``warning``: int counts of tagged lines.
    - ``has_error_level``: True if any Fatal or Severe present.
    - ``fatal_lines`` / ``severe_lines``: list[str] of the **full original
      line text** (with the ``** Severity **`` tag stripped to a clean
      message), so an LLM can read what actually went wrong.

    Missing / unreadable file -> zero counts, empty lists, no exception.
    """
    result: dict[str, Any] = {
        "fatal": 0,
        "severe": 0,
        "warning": 0,
        "has_error_level": False,
        "fatal_lines": [],
        "severe_lines": [],
    }
    if err_path is None:
        return result
    err_path = Path(err_path)
    if not err_path.exists():
        return result
    try:
        text = err_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    for line in text.splitlines():
        m = _SEVERITY_RE.match(line)
        if not m:
            continue
        level = m.group(1).lower()
        # Strip the "** Severity **" tag prefix to give the LLM a clean
        # message (everything after the closing "**" of the tag).
        message = line[m.end() :].strip()
        if level == "fatal":
            result["fatal"] += 1
            result["fatal_lines"].append(message)
        elif level == "severe":
            result["severe"] += 1
            result["severe_lines"].append(message)
        elif level == "warning":
            result["warning"] += 1

    result["has_error_level"] = (result["fatal"] + result["severe"]) > 0
    return result


def has_error_level(err_path: Path | str | None) -> bool:
    """True if ``eplusout.err`` contains any Fatal or Severe line."""
    return extract_errors(err_path)["has_error_level"]


def format_errors_for_llm(err_info: dict[str, Any], max_lines: int = 20) -> str:
    """Render Fatal/Severe errors as a compact text block for an LLM prompt.

    Lists Fatal first (most critical), then Severe, capped at ``max_lines``
    total to bound token usage. Returns an empty string if there are none.
    """
    fatal = err_info.get("fatal_lines", [])
    severe = err_info.get("severe_lines", [])
    if not fatal and not severe:
        return ""
    lines: list[str] = []
    if fatal:
        lines.append(f"Fatal errors ({len(fatal)}):")
        for msg in fatal[:max_lines]:
            lines.append(f"  - {msg}")
    remaining = max(0, max_lines - len(fatal[:max_lines]))
    if severe and remaining > 0:
        lines.append(f"Severe errors ({len(severe)}):")
        for msg in severe[:remaining]:
            lines.append(f"  - {msg}")
    return "\n".join(lines)
