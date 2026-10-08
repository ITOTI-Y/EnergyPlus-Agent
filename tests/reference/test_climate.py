import math
from pathlib import Path

import pytest

from src.reference.climate import from_epw

WEATHER = Path(__file__).parents[2] / "data" / "weather"


def _epw(tmp_path: Path, annual_mean: float, rain_mm: float | None) -> Path:
    """An EPW with a 10 K seasonal swing around the mean and even rain."""
    header = "\n".join(f"HEADER{i}" for i in range(8))
    rows = []
    for hour in range(8760):
        month = min(hour // 730 + 1, 12)
        day = (hour // 24) % 28 + 1
        fields = ["1999", str(month), str(day), str(hour % 24 + 1), "0", "src"]
        swing = 10.0 * math.cos(2 * math.pi * (hour / 8760 - 0.55))
        fields += [f"{annual_mean + swing:.2f}"] + ["0"] * 26
        fields += ["999" if rain_mm is None else str(rain_mm / 8760), "0"]
        rows.append(",".join(fields))
    path = tmp_path / "site.epw"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_shenzhen_is_1a_and_close_enough_to_2a_to_take_it_too():
    climate = from_epw(WEATHER / "Shenzhen.epw")

    assert climate.cdd10 == pytest.approx(5153, abs=1)
    assert climate.zone == "1A"
    assert climate.candidates == ["1A", "2A"]


@pytest.mark.parametrize(
    ("temperature", "rain", "zone", "candidates"),
    [
        (16.0, 1500.0, "3A", ["3A"]),  # HDD18 about 1550, CDD10 about 2470
        (16.0, 200.0, "3B", ["3B"]),  # 200 mm < 2 x (16 + 7) cm
        (16.0, None, "3", ["3A", "3B", "3C"]),  # rain unknown: all regimes
    ],
)
def test_zone_needs_rain_for_the_moisture_regime(
    tmp_path, temperature, rain, zone, candidates
):
    climate = from_epw(_epw(tmp_path, temperature, rain))

    assert climate.zone == zone
    assert climate.candidates == candidates
