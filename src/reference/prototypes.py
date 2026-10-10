"""DOE prototype building models: download, reduce, upgrade to 26.1, label.

The commercial (ASHRAE 90.1) and residential (IECC) prototypes published by
PNNL on energycodes.gov ship as zips of IDF files for EnergyPlus 22.1 and
23.1. Only materials, constructions and schedules are kept, then upgraded
to 26.1 so idfpy reads them with the current field layouts.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal
from urllib.request import Request, urlopen

from loguru import logger

BASE_URL: Final = "https://www.energycodes.gov/sites/default/files"
# energycodes.gov answers 404 to clients without a browser user agent.
USER_AGENT: Final = "Mozilla/5.0"

COMMERCIAL_TYPES: Final = (
    "ApartmentHighRise",
    "ApartmentMidRise",
    "Hospital",
    "HotelLarge",
    "HotelSmall",
    "OfficeLarge",
    "OfficeMedium",
    "OfficeSmall",
    "OutPatientHealthCare",
    "RestaurantFastFood",
    "RestaurantSitDown",
    "RetailStandalone",
    "RetailStripmall",
    "SchoolPrimary",
    "SchoolSecondary",
    "Warehouse",
)
COMMERCIAL_EDITIONS: Final = (2019, 2022)
RESIDENTIAL_ZONES: Final = (
    "1A", "2A", "2B", "3A", "3B", "3C", "4A", "4B",
    "4C", "5A", "5B", "5C", "6A", "6B", "7", "8",
)  # fmt: skip
RESIDENTIAL_EDITIONS: Final = (2021, 2024)
TARGET_VERSION: Final = "26.1"

type Category = Literal["commercial", "residential"]


@dataclass(frozen=True, slots=True)
class PrototypeModel:
    """One prototype IDF and what it represents."""

    category: Category
    building_type: str
    standard: str
    climate_zone: str
    location: str
    path: Path
    roles: dict[str, set[str]]
    """Where each construction is used, by lower-case construction name."""


def archive_urls() -> list[str]:
    """Zips holding the selected prototypes.

    Residential IECC 2021 models ship in one zip per climate zone together
    with the older editions; IECC 2024 has a zip of its own.
    """
    commercial = [
        f"{BASE_URL}/2023-10/ASHRAE901_{kind}_STD{year}.zip"
        for kind in COMMERCIAL_TYPES
        for year in COMMERCIAL_EDITIONS
    ]
    residential = [
        f"{BASE_URL}/2024-12/resstd_CZ{zone}{suffix}.zip"
        for zone in RESIDENTIAL_ZONES
        for suffix in ("", "_IECC_2024")
    ]
    return commercial + residential


def download(urls: list[str], directory: Path) -> list[Path]:
    """Download each zip unless it is already there.

    Raises:
        urllib.error.URLError: If a download fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in urls:
        path = directory / url.rsplit("/", 1)[1]
        if not path.exists():
            logger.info("Downloading {}", url)
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as reply:
                partial = path.with_suffix(".part")
                partial.write_bytes(reply.read())
                partial.rename(path)
        paths.append(path)
    return paths


_COMMERCIAL: Final = re.compile(r"ASHRAE901_(\w+?)_STD(\d{4})_(\w+)\.idf")
_RESIDENTIAL: Final = re.compile(
    r"US\+(SF|MF)\+CZ(\d[ABC]?)\w*\+([\w-]+)\+(\w+)\+IECC_(\d{4})\.idf"
)
# Residential models repeat each envelope for four heating systems and
# several water heaters; one heating system keeps every envelope.
RESIDENTIAL_HEATING: Final = "hp"
# The zone of the location's climate; CZ_Label is the zone whose 90.1
# requirements apply, e.g. 1A for Ho Chi Minh City in zone 0A.
_ZONE_LABEL: Final = re.compile(r"^!\s*AnalysisClimateZone\s*=\s*(\w+)", re.MULTILINE)
_RESIDENTIAL_TYPES: Final = {"SF": "SingleFamily", "MF": "MultiFamily"}


def _label(member: str, text: str) -> tuple[Category, str, str, str, str] | None:
    """Category, building type, standard, climate zone and location of a member.

    Residential models carry no location; their climate zone names it.
    """
    name = Path(member).name
    if match := _COMMERCIAL.fullmatch(name):
        kind, year, city = match.groups()
        zone = _ZONE_LABEL.search(text)
        if zone is None:
            raise ValueError(f"{member}: no AnalysisClimateZone in the header")
        return "commercial", kind, f"ASHRAE 90.1-{year}", zone[1], city
    if match := _RESIDENTIAL.fullmatch(name):
        kind, zone, heating, foundation, year = match.groups()
        if int(year) not in RESIDENTIAL_EDITIONS or heating != RESIDENTIAL_HEATING:
            return None
        building = f"{_RESIDENTIAL_TYPES[kind]}_{foundation}"
        return "residential", building, f"IECC {year}", zone, f"CZ{zone}"
    return None


# Objects the reference library reads.
KEPT_TYPES: Final = frozenset(
    t.lower()
    for t in (
        "Version",
        "Material",
        "Material:NoMass",
        "Material:AirGap",
        "WindowMaterial:SimpleGlazingSystem",
        "WindowMaterial:Glazing",
        "WindowMaterial:Gas",
        "Construction",
        "ScheduleTypeLimits",
        "Schedule:Compact",
        "Schedule:Year",
        "Schedule:Week:Compact",
        "Schedule:Day:Hourly",
    )
)


def reduce_model(text: str) -> str:
    """The model with only KEPT_TYPES objects, comments dropped.

    Surfaces are left out: where constructions are used is read from the
    original text by construction_roles.
    """
    body = re.sub(r"!.*", "", text)
    kept = []
    for obj in body.split(";"):
        fields = obj.strip()
        if fields and fields.split(",", 1)[0].strip().lower() in KEPT_TYPES:
            kept.append(fields + ";")
    return "\n\n".join(kept) + "\n"


def construction_roles(text: str) -> dict[str, set[str]]:
    """Where each construction is used, e.g. 'Wall, Outdoors' or 'Window'.

    Read from the original model, whose surfaces are not upgraded; the
    fields read keep their positions from EnergyPlus 9.6 on: surface type,
    construction, and for base surfaces the boundary after zone and space.
    InternalMass names its construction right after its own name.
    """
    roles: dict[str, set[str]] = {}
    for obj in re.sub(r"!.*", "", text).split(";"):
        fields = [f.strip() for f in obj.split(",")]
        kind = fields[0].lower()
        if kind == "buildingsurface:detailed" and len(fields) > 6:
            role = f"{fields[2]}, {fields[6]}"
        elif kind == "fenestrationsurface:detailed" and len(fields) > 3:
            role = fields[2]
        elif kind == "internalmass" and len(fields) > 2:
            roles.setdefault(fields[2].lower(), set()).add("InternalMass")
            continue
        else:
            continue
        roles.setdefault(fields[3].lower(), set()).add(role)
    return roles


def extract(archives: list[Path], directory: Path) -> list[PrototypeModel]:
    """Unpack the selected IDFs of each archive, reduced and upgraded.

    Of models with the same label, which differ only in the water heater,
    the first in name order is kept.
    """
    directory.mkdir(parents=True, exist_ok=True)
    models = []
    seen: set[tuple[str, ...]] = set()
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            for member in sorted(bundle.namelist()):
                if not member.lower().endswith(".idf"):
                    continue
                text = bundle.read(member).decode("utf-8", errors="replace")
                if (label := _label(member, text)) is None or label in seen:
                    continue
                seen.add(label)
                path = directory / Path(member).name
                path.write_text(upgrade(reduce_model(text)), encoding="utf-8")
                models.append(
                    PrototypeModel(*label, path=path, roles=construction_roles(text))
                )
    return models


_VERSION: Final = re.compile(r"(Version\s*,\s*)(\d+)\.(\d+)[\d.]*", re.I)
UPGRADABLE_FROM: Final = (22, 1)


def upgrade(text: str) -> str:
    """A reduced model upgraded to TARGET_VERSION.

    The transition rules of EnergyPlus 22.1 to 26.1 (Rules*.md of the
    IDFVersionUpdater) change none of KEPT_TYPES, so upgrading them only
    sets the version. Running the transition programs on reduced models
    gave identical objects; those programs take minutes per full model.

    Raises:
        ValueError: If the model has no Version or predates 22.1, whose
            transitions do change these objects.
    """
    match = _VERSION.search(text)
    if match is None:
        raise ValueError("no Version object")
    if (int(match[2]), int(match[3])) < UPGRADABLE_FROM:
        raise ValueError(f"version {match[2]}.{match[3]} predates 22.1")
    return text[: match.start()] + match[1] + TARGET_VERSION + text[match.end() :]
