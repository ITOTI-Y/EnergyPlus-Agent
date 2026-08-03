"""Generate the text-only test-case corpus for the robustness benchmark.

Creates 9 scale-buckets x 10 cases = 90 ``testdata_prompt.json`` files under

    tests/agent/test_data/text_only/<category>/<scale>/case_<NN>/

Every case is built from real, literature-grounded parameters (DOE
Commercial Reference Buildings, PNNL prototype models, NIST TN 1765) and
carries a detailed free-text ``Description`` so the agent has enough
information to build geometry without drawings (text-only mode).

Idempotent — existing files are overwritten. Run from the repo root:

    uv run python -m tests.agent.gen_test_data
"""

import json
import math
from pathlib import Path
from typing import Any, Final

from loguru import logger

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
OUT_ROOT: Final = REPO_ROOT / "tests" / "agent" / "test_data" / "text_only"

CASES_PER_BUCKET: Final = 10

# Parameter tables grounded in DOE / NIST prototypes.
# category -> scale -> params; areas/zones are per-floor unless noted,
# floors & total zones derived.
BUCKETS: Final[dict[str, dict[str, dict[str, Any]]]] = {
    "residential": {
        # small: single-unit dwellings (studio / 1-2BR). Simplest geometry.
        "small": {
            "area_range": (45, 110),  # m^2 — one unit's footprint
            "floors": (1, 1),
            "units_per_floor": (1, 1),
            "zones_per_unit": (1, 2),
            "height": (2.8, 3.0),
            "wwr_range": (0.15, 0.22),
            "city": "Shenzhen",
            "type_label": "Small apartment / studio unit",
            "rooms": [
                "open-plan studio: combined living/sleeping + kitchenette + bathroom",
                "1-bedroom: living room + bedroom + kitchen + bathroom",
                "1-bedroom: studio living + separate bedroom + bathroom + balcony",
                "2-bedroom: living + dining + kitchen + 2 bedrooms + 1 bathroom",
                "loft: open living/kitchen below, sleeping mezzanine + bathroom",
            ],
        },
        # medium: low-rise multifamily (3 storeys, several stacked units)
        # with multiple dwellings and a shared core / circulation.
        "medium": {
            "area_range": (600, 1200),  # total building area
            "floors": (3, 3),
            "units_per_floor": (2, 4),
            "zones_per_unit": (1, 2),
            "height": (2.9, 3.1),
            "wwr_range": (0.18, 0.28),
            "city": "Guangzhou",
            "type_label": "Low-rise multifamily apartment building",
            "rooms": [
                "per floor: 2 units (2B+1B each) + shared stairwell core",
                "per floor: 3 units (1B/2B/2B) + elevator lobby + stair core",
                "per floor: 2 corner units (3B+2B) + central corridor + stair",
                "per floor: 4 compact units (1B each) + double-loaded corridor + core",
                "per floor: 2 units (2B+1B) + shared lobby + stair + utility",
            ],
        },
        # large: mid/high-rise apartment tower (10+ storeys, many units per
        # floor). Most complex: vertical stacking, core + corridor.
        "large": {
            "area_range": (8000, 20000),  # total building area
            "floors": (10, 18),
            "units_per_floor": (4, 8),
            "zones_per_unit": (1, 2),
            "height": (3.0, 3.2),
            "wwr_range": (0.25, 0.40),
            "city": "Shenzhen",
            "type_label": "High-rise apartment tower",
            "rooms": [
                "per typical floor: 6 units (mix of 1B-3B) + central core "
                "(2 lifts + stairs + lobby)",
                "per typical floor: 8 compact units along double-loaded corridor "
                "+ core",
                "per typical floor: 4 corner units (3B-4B) + 2 lifts + stair "
                "+ service core",
                "per typical floor: 6 units + sky lobby + 2 lifts + stair + MEP shaft",
                "per typical floor: 5 units (2B/3B) + elevator lobby + stair "
                "+ garbage room",
            ],
        },
    },
    "office": {
        "small": {
            "area_range": (300, 600),
            "floors": (1, 2),
            "zones_per_floor": (4, 6),
            "height": (3.2, 3.6),
            "wwr_range": (0.25, 0.35),
            "city": "Shenzhen",
            "type_label": "Small office building",
            "rooms": [
                "4 perimeter offices + central core (corridor + restroom "
                "+ server closet)",
                "open-plan office + 3 private offices + meeting room + restroom",
                "reception + open office + 2 meeting rooms + break room + restroom",
                "co-working space + 3 private offices + pantry + restroom",
                "open office + manager office + 2 meeting rooms + IT closet + restroom",
            ],
        },
        "medium": {
            "area_range": (2000, 5000),
            "floors": (3, 3),
            "zones_per_floor": (4, 6),
            "height": (3.4, 3.8),
            "wwr_range": (0.30, 0.40),
            "city": "Guangzhou",
            "type_label": "Medium office building",
            "rooms": [
                "per floor: 4 perimeter office zones (N/E/S/W) + central core",
                "per floor: open-plan office + 2 meeting rooms + core + server room",
                "per floor: open office + executive suite + meeting + core",
                "per floor: 4 open-plan bays + core (restroom + IT + circulation)",
                "per floor: open office + 3 private offices + core",
            ],
        },
        "large": {
            "area_range": (10000, 25000),
            "floors": (6, 12),
            "zones_per_floor": (4, 8),
            "height": (3.6, 4.0),
            "wwr_range": (0.35, 0.50),
            "city": "Shenzhen",
            "type_label": "Large office / corporate headquarters",
            "rooms": [
                "typical floor: 4 perimeter zones + core; lobby on ground, "
                "data center on basement",
                "typical floor: open office + meeting rooms + core; ground "
                "retail/lobby, top executive floor",
                "typical floor: open-plan + private offices + core; ground "
                "lobby, rooftop plant",
                "typical floor: 4 open bays + core; podium retail (ground-2), "
                "tower offices (3-top)",
                "typical floor: open office + meeting suite + core; ground "
                "lobby, mid mechanical floor",
            ],
        },
    },
    "retail": {
        "small": {
            "area_range": (500, 1500),
            "floors": (1, 1),
            "zones_per_floor": (3, 6),
            "height": (3.6, 4.2),
            "wwr_range": (0.30, 0.45),
            "city": "Shenzhen",
            "type_label": "Stand-alone retail / convenience store",
            "rooms": [
                "retail sales floor + back storage + staff room + restroom",
                "retail area + storeroom + office + restroom",
                "sales floor + storage + cold room + restroom",
                "showroom + stockroom + office + restroom",
                "retail floor + warehouse + staff break + restroom",
            ],
        },
        "medium": {
            "area_range": (2000, 5000),
            "floors": (1, 2),
            "zones_per_floor": (6, 12),
            "height": (4.0, 5.0),
            "wwr_range": (0.35, 0.50),
            "city": "Guangzhou",
            "type_label": "Strip mall / supermarket",
            "rooms": [
                "5-6 retail units along a covered walkway + shared storage "
                "+ restroom core",
                "supermarket sales floor + cold storage + dry storage + offices "
                "+ restroom",
                "6 shop units + food court + common area + restrooms + loading bay",
                "ground: 5 retail units + storage; upper: offices + staff area",
                "department store sales + stockrooms + offices + customer restroom",
            ],
        },
        "large": {
            "area_range": (8000, 20000),
            "floors": (2, 4),
            "zones_per_floor": (6, 10),
            "height": (4.5, 6.0),
            "wwr_range": (0.30, 0.45),
            "city": "Shenzhen",
            "type_label": "Large shopping mall",
            "rooms": [
                "per floor: anchor store + multiple shop units + central atrium "
                "+ food court + restrooms",
                "per floor: retail concourse + specialty shops + cinema (top) "
                "+ dining + restrooms",
                "per floor: department store + boutiques + circulation mall "
                "+ services + restrooms",
                "per floor: hypermarket (ground) + retail units + entertainment "
                "zone + food court + restrooms",
                "per floor: anchor shops + mid-mall retail + atrium + dining "
                "terrace + restrooms",
            ],
        },
    },
}

# Orientations rotated through the cases to add variety.
ORIENTATIONS: Final = [0, 90, 180, 270, 45, 135, 225, 315, 15, 75]

# Footprint aspect ratios (W:D) to vary geometry per case.
ASPECTS: Final = [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (1.2, 1.0), (1.8, 1.0)]


def _mid(bounds: tuple[float, float]) -> float:
    return round((bounds[0] + bounds[1]) / 2, 1)


def build_case(
    category: str, scale: str, params: dict[str, Any], idx: int
) -> dict[str, Any]:
    """Build one testdata_prompt dict for a (category, scale, idx).

    Supports two zone-count schemas: multifamily (residential) uses
    ``units_per_floor`` * ``zones_per_unit``; office/retail use
    ``zones_per_floor`` directly.
    """
    floors = _mid(params["floors"])
    area_total = _mid(params["area_range"])
    per_floor = round(area_total / floors, 1)
    height = _mid(params["height"])
    wwr = _mid(params["wwr_range"])
    orientation = ORIENTATIONS[idx % len(ORIENTATIONS)]
    w, d = ASPECTS[idx % len(ASPECTS)]
    rooms = params["rooms"][idx % len(params["rooms"])]
    city = params["city"]
    btype = params["type_label"]

    is_multifamily = "units_per_floor" in params
    if is_multifamily:
        upf = round(_mid(params["units_per_floor"]))
        zpu = round(_mid(params["zones_per_unit"]))
        zpf = upf * zpu
        total_zones = round(floors * zpf)
    else:
        upf = None
        zpf = round(_mid(params["zones_per_floor"]))
        total_zones = round(floors * zpf)

    # Footprint from per-floor area & aspect ratio:
    # area = W * D, W/D = aspect  ->  W = sqrt(area*aspect), D = sqrt(area/aspect)
    width = round(math.sqrt(per_floor * w / d), 1)
    depth = round(math.sqrt(per_floor * d / w), 1)

    name = f"{category}_{scale}_{idx + 1:02d}"

    # Multifamily-specific clause: surface the dwelling-unit count so the
    # agent creates stacked, repeated dwellings, not one big zone.
    unit_clause = ""
    if is_multifamily and upf:
        unit_clause = (
            f" There are {upf} dwelling units per floor "
            f"({total_zones // max(int(floors), 1) // upf} thermal zone(s) per unit), "
            f"so model each unit as its own zone(s) stacked vertically across "
            f"the {int(floors)} floor(s), plus a shared core/circulation."
        )

    description = (
        f"A {btype.lower()} located in {city}, China. "
        f"The building footprint is approximately {width} m (east-west) by "
        f"{depth} m (north-south), rectangular, with the long axis rotated "
        f"{orientation} degrees from north. Total gross floor area is about "
        f"{area_total} m^2 spread over {int(floors)} floor(s), giving roughly "
        f"{per_floor} m^2 per floor. Floor-to-floor height is {height} m. "
        f"The building is divided into {total_zones} thermal zones in total "
        f"(about {zpf} per floor).{unit_clause} Typical space layout: {rooms}. "
        f"Window-to-wall ratio is approximately {wwr:.0%} on exterior walls, "
        f"with windows distributed on all four facades. "
        f"Model all zones with appropriate constructions, glazing, occupancy, "
        f"lighting, equipment and HVAC (ideal loads) so the IDF runs in "
        f"EnergyPlus without errors."
    )

    data: dict[str, Any] = {
        "TestName": name,
        "Building location": city,
        "Building type": btype,
        "Floor area": f"{area_total}m2",
        "Number of floors": str(int(floors)),
        "Number of thermal zones per floor of the building": str(zpf),
        "Number of total thermal zones in the building": str(total_zones),
        "Footprint width (east-west, m)": str(width),
        "Footprint depth (north-south, m)": str(depth),
        "Floor-to-floor height (m)": str(height),
        "Orientation (degrees from north)": str(orientation),
        "Window-to-wall ratio": f"{wwr:.2f}",
        "Space layout": rooms,
        "Description": description,
        # Image fields left empty (text-only mode).
        "Top view path of the building": "",
        "Front view path of the building": "",
        "Building side view path": "",
        "Path of the supplementary plan example drawing for the building": "",
    }
    if is_multifamily:
        data["Dwelling units per floor"] = str(upf)
    return data


def main() -> None:
    """Write every bucket's cases and log per-bucket counts."""
    total = 0
    for category, scales in BUCKETS.items():
        for scale, params in scales.items():
            for idx in range(1, CASES_PER_BUCKET + 1):
                case_dir = OUT_ROOT / category / scale / f"case_{idx:02d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                data = build_case(category, scale, params, idx - 1)
                (case_dir / "testdata_prompt.json").write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                total += 1
    logger.info("generated {} cases under {}", total, OUT_ROOT)
    for category, scales in BUCKETS.items():
        for scale in scales:
            n = len(list((OUT_ROOT / category / scale).glob("case_*")))
            logger.info("  {}/{}: {} cases", category, scale, n)


if __name__ == "__main__":
    main()
