"""ASHRAE 169 climate zone of a weather file.

The thermal zone follows from degree days of the daily mean dry-bulb
temperature. Thresholds are the SI definitions of ASHRAE 169-2013, which
IECC 2021 Table R301.3(2) restates in IP units: zone 0 above 6000 CDD10,
1 above 5000, 2 above 3500, then by HDD18 up to 2000 (3), 3000 (4),
4000 (5), 5000 (6), 7000 (7), above that 8. The moisture regime (A humid,
B dry, C marine) needs precipitation, which typical-year weather files
often leave empty; it is then left open rather than guessed.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Final

_CDD10_LIMITS: Final = ((6000.0, "0"), (5000.0, "1"), (3500.0, "2"))
_HDD18_LIMITS: Final = (
    (2000.0, "3"),
    (3000.0, "4"),
    (4000.0, "5"),
    (5000.0, "6"),
    (7000.0, "7"),
)
NEAR_BOUNDARY: Final = 0.1
"""Relative distance to a threshold within which the neighbour zone also counts."""
_MISSING_PRECIPITATION: Final = 999.0
_LETTERS: Final = {"0": "AB", "1": "AB", "2": "AB", "3": "ABC", "4": "ABC",
                   "5": "ABC", "6": "AB", "7": "", "8": ""}  # fmt: skip


@dataclass(frozen=True, slots=True)
class Climate:
    cdd10: float
    hdd18: float
    zone: str
    """Thermal zone number, with the moisture letter when it is known."""
    candidates: list[str]
    """Zones whose reference objects suit the location, e.g. ['1A', '2A']."""


def _thermal(cdd10: float, hdd18: float) -> str:
    for limit, zone in _CDD10_LIMITS:
        if cdd10 > limit:
            return zone
    for limit, zone in _HDD18_LIMITS:
        if hdd18 <= limit:
            return zone
    return "8"


def _neighbours(cdd10: float, hdd18: float) -> set[str]:
    """Thermal zones of the location and of thresholds it lies close to."""
    zones = {_thermal(cdd10, hdd18)}
    for value, limits, is_cdd in (
        (cdd10, _CDD10_LIMITS, True),
        (hdd18, _HDD18_LIMITS, False),
    ):
        for limit, _ in limits:
            if abs(value - limit) <= NEAR_BOUNDARY * limit:
                shift = (1.0 + NEAR_BOUNDARY) * limit, (1.0 - NEAR_BOUNDARY) * limit
                for nudged in shift:
                    zones.add(
                        _thermal(nudged, hdd18) if is_cdd else _thermal(cdd10, nudged)
                    )
    return zones


def _moisture(
    zone: str, monthly_temperature: list[float], annual_temperature: float,
    precipitation_mm: float | None,
) -> str | None:  # fmt: skip
    """'A', 'B', 'C', or None when precipitation is unknown.

    Marine: coldest month between -3 and 18 C, warmest below 22 C, and at
    least four months above 10 C; the seasonal rainfall condition of the
    definition is not checked. Dry: annual precipitation below
    2 x (T + 7) cm.
    """
    letters = _LETTERS[zone]
    if not letters:
        return None
    if (
        "C" in letters
        and -3.0 < min(monthly_temperature) < 18.0
        and max(monthly_temperature) < 22.0
        and sum(t > 10.0 for t in monthly_temperature) >= 4
    ):
        return "C"
    if precipitation_mm is None:
        return None
    return "B" if precipitation_mm / 10.0 < 2.0 * (annual_temperature + 7.0) else "A"


def from_epw(path: Path) -> Climate:
    """Climate zone of an EPW file.

    Raises:
        ValueError: If the file does not hold 8760 or 8784 hourly records.
    """
    with path.open(encoding="utf-8", errors="replace") as f:
        rows = [r for r in csv.reader(f) if len(r) > 33 and r[0].isdigit()]
    if len(rows) not in (8760, 8784):
        raise ValueError(f"{path}: {len(rows)} hourly records")
    daily: dict[tuple[int, int], list[float]] = {}
    precipitation = 0.0
    precipitation_known = False
    for row in rows:
        daily.setdefault((int(row[1]), int(row[2])), []).append(float(row[6]))
        depth = float(row[33])
        if depth < _MISSING_PRECIPITATION:
            precipitation += depth
            precipitation_known = precipitation_known or depth > 0
    days = {key: mean(values) for key, values in daily.items()}
    cdd10 = sum(max(0.0, t - 10.0) for t in days.values())
    hdd18 = sum(max(0.0, 18.0 - t) for t in days.values())
    monthly = [
        mean(t for (month, _), t in days.items() if month == m) for m in range(1, 13)
    ]
    annual = mean(days.values())
    numbers = _neighbours(cdd10, hdd18)
    thermal = _thermal(cdd10, hdd18)
    candidates = []
    for number in sorted(numbers):
        letter = _moisture(
            number, monthly, annual, precipitation if precipitation_known else None
        )
        letters = letter or _LETTERS[number]
        candidates += [number + c for c in letters] or [number]
    own = _moisture(
        thermal, monthly, annual, precipitation if precipitation_known else None
    )
    return Climate(cdd10, hdd18, thermal + (own or ""), candidates)
