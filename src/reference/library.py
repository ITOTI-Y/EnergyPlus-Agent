"""The reference library: deduplicated prototype objects in one SQLite file.

Objects with the same type, name and field values are merged, keeping every
building type, standard, climate zone and use they appear with. Each object
gets a description for semantic search; embeddings are stored in the same
file so a vector index can be rebuilt without embedding again.
"""

import hashlib
import json
import sqlite3
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from idfpy import IDF
from loguru import logger

from src.reference.extract import Kind, ReferenceObject, extract_objects
from src.reference.prototypes import PrototypeModel

SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS reference_object (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    object_type TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    description TEXT NOT NULL,
    categories TEXT NOT NULL,
    building_types TEXT NOT NULL,
    standards TEXT NOT NULL,
    climate_zones TEXT NOT NULL,
    roles TEXT NOT NULL,
    model_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding (
    id TEXT PRIMARY KEY REFERENCES reference_object(id),
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# ASHRAE 169 climate zone names, with Chinese wording for bilingual queries.
_THERMAL: Final = {
    "0": "extremely hot 极热",
    "1": "very hot 炎热",
    "2": "hot 热",
    "3": "warm 温暖",
    "4": "mixed 温和",
    "5": "cool 凉爽",
    "6": "cold 寒冷",
    "7": "very cold 严寒",
    "8": "subarctic 亚北极",
}
_MOISTURE: Final = {"A": "humid 潮湿", "B": "dry 干燥", "C": "marine 海洋性"}


def zone_words(zone: str) -> str:
    """'1A' -> '1A very hot 炎热, humid 潮湿'."""
    words = [_THERMAL[zone[0]]]
    if len(zone) > 1:
        words.append(_MOISTURE[zone[1]])
    return f"{zone} {', '.join(words)}"


@dataclass(slots=True)
class LibraryEntry:
    """A merged object and every model context it appears in."""

    id: str
    kind: Kind
    object_type: str
    name: str
    data: dict[str, Any]
    categories: set[str] = field(default_factory=set)
    building_types: set[str] = field(default_factory=set)
    standards: set[str] = field(default_factory=set)
    climate_zones: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    model_count: int = 0
    description: str = ""


def entry_id(obj: ReferenceObject) -> str:
    canonical = json.dumps(
        [obj.kind, obj.object_type, obj.data], sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


@dataclass(slots=True)
class BuildReport:
    models: int
    entries: dict[str, int]
    skipped_schedules: list[str]


def collect(models: Iterable[PrototypeModel]) -> tuple[list[LibraryEntry], list[str]]:
    """Merge the objects of all models; returns entries and skipped schedules."""
    entries: dict[str, LibraryEntry] = {}
    skipped: set[str] = set()
    for model in models:
        extraction = extract_objects(IDF.load(model.path), model.roles)
        skipped.update(extraction.skipped)
        for obj in extraction.objects:
            key = entry_id(obj)
            entry = entries.setdefault(
                key, LibraryEntry(key, obj.kind, obj.object_type, obj.name, obj.data)
            )
            entry.categories.add(model.category)
            entry.building_types.add(model.building_type)
            entry.standards.add(model.standard)
            entry.climate_zones.add(model.climate_zone)
            entry.roles |= obj.roles
            entry.model_count += 1
    return list(entries.values()), sorted(skipped)


def _number(value: Any, unit: str) -> str:
    return f"{value:g} {unit}" if isinstance(value, int | float) else ""


def _material_summary(data: dict[str, Any], object_type: str) -> str:
    match object_type:
        case "Material":
            parts = [
                _number(data.get("thickness"), "m thick"),
                _number(data.get("conductivity"), "W/m-K"),
                _number(data.get("density"), "kg/m3"),
                _number(data.get("specific_heat"), "J/kg-K"),
            ]
        case "Material:NoMass" | "Material:AirGap":
            parts = [_number(data.get("thermal_resistance"), "m2-K/W")]
        case "WindowMaterial:SimpleGlazingSystem":
            parts = [
                "U-factor " + _number(data.get("u_factor"), "W/m2-K"),
                f"SHGC {data.get('solar_heat_gain_coefficient')}",
                f"visible transmittance {data.get('visible_transmittance', 'n/a')}",
            ]
        case "WindowMaterial:Glazing":
            parts = [
                _number(data.get("thickness"), "m thick"),
                f"solar transmittance {data.get('solar_transmittance_at_normal_incidence')}",
            ]
        case _:
            parts = [str(data.get("gas_type", "")), _number(data.get("thickness"), "m")]
    return ", ".join(p for p in parts if p)


def layer_resistance(material: dict[str, Any]) -> float | None:
    """Thermal resistance of one opaque layer, None for window layers."""
    match material["object_type"]:
        case "Material":
            return material["thickness"] / material["conductivity"]
        case "Material:NoMass" | "Material:AirGap":
            return material["thermal_resistance"]
        case _:
            return None


def _construction_summary(data: dict[str, Any]) -> str:
    layers = data["materials"]
    named = "; ".join(
        f"{m['name']} ({_material_summary(m, m['object_type'])})" for m in layers
    )
    resistances = [layer_resistance(m) for m in layers]
    if all(r is not None for r in resistances):
        total = sum(r for r in resistances if r is not None)
        return (
            f"layers outside to inside: {named}. Layer resistance {total:.2f} "
            f"m2-K/W, U-value without surface films {1 / total:.2f} W/m2-K"
        )
    return f"window layers outside to inside: {named}"


def _schedule_summary(data: dict[str, Any]) -> str:
    limits = data.get("schedule_type_limits") or {}
    blocks = []
    for period in data["periods"]:
        days = ", ".join(
            f"{day['For']}: "
            + " ".join(
                f"{t['Until']['Value']:g} until {t['Until']['Time']}"
                for t in day["Times"]
            )
            for day in period["Days"]
        )
        blocks.append(f"through {period['Through']}: {days}")
    kind = limits.get("unit_type") or limits.get("name") or "values"
    return f"{kind} schedule; " + " | ".join(blocks)


@dataclass(frozen=True, slots=True)
class Coverage:
    """Building types and climate zones present in the source models."""

    building_types: dict[str, set[str]]
    climate_zones: dict[str, set[str]]

    @classmethod
    def of(cls, models: Iterable[PrototypeModel]) -> "Coverage":
        types: dict[str, set[str]] = {}
        zones: dict[str, set[str]] = {}
        for model in models:
            types.setdefault(model.category, set()).add(model.building_type)
            zones.setdefault(model.category, set()).add(model.climate_zone)
        return cls(types, zones)


def _span(values: set[str], universe: set[str], everything: str) -> str:
    return everything if values >= universe else ", ".join(sorted(values))


def describe(entry: LibraryEntry, coverage: Coverage) -> str:
    """Text embedded for semantic search.

    Building types and climate zones are named only when the object is
    specific to some of them, so the text of an object used everywhere is
    about the object itself.
    """
    match entry.kind:
        case "material":
            what = f"{entry.object_type} '{entry.name}': " + _material_summary(
                entry.data, entry.object_type
            )
        case "construction":
            what = f"Construction '{entry.name}': " + _construction_summary(entry.data)
        case "schedule":
            what = f"Schedule '{entry.name}': " + _schedule_summary(entry.data)
    all_types = set().union(*(coverage.building_types[c] for c in entry.categories))
    all_zones = set().union(*(coverage.climate_zones[c] for c in entry.categories))
    categories = " and ".join(sorted(entry.categories))
    types = _span(entry.building_types, all_types, f"all {categories} building types")
    zones = (
        "all climate zones"
        if entry.climate_zones >= all_zones
        else "; ".join(zone_words(z) for z in sorted(entry.climate_zones))
    )
    used = f" Used as {', '.join(sorted(entry.roles))}." if entry.roles else ""
    return (
        f"{what}.{used} From DOE {categories} prototype buildings "
        f"({', '.join(sorted(entry.standards))}): {types}. Climate zones: {zones}."
    )


def write(path: Path, entries: list[LibraryEntry], metadata: dict[str, Any]) -> None:
    """Write a new library file, replacing any existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO reference_object VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    e.id,
                    e.kind,
                    e.object_type,
                    e.name,
                    json.dumps(e.data, default=str),
                    e.description,
                    json.dumps(sorted(e.categories)),
                    json.dumps(sorted(e.building_types)),
                    json.dumps(sorted(e.standards)),
                    json.dumps(sorted(e.climate_zones)),
                    json.dumps(sorted(e.roles)),
                    e.model_count,
                )
                for e in entries
            ],
        )
        db.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in metadata.items()],
        )


def build(
    models: list[PrototypeModel], path: Path, metadata: dict[str, Any]
) -> BuildReport:
    """Extract, merge and describe all objects, then write the library."""
    entries, skipped = collect(models)
    coverage = Coverage.of(models)
    for entry in entries:
        entry.description = describe(entry, coverage)
    write(path, entries, {**metadata, "models": len(models)})
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
    logger.info("Library {}: {} entries {}", path, len(entries), counts)
    return BuildReport(len(models), counts, skipped)


def pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def store_embeddings(path: Path, vectors: dict[str, list[float]]) -> None:
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT OR REPLACE INTO embedding VALUES (?, ?)",
            [(key, pack(v)) for key, v in vectors.items()],
        )


def set_metadata(path: Path, values: dict[str, Any]) -> None:
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT OR REPLACE INTO metadata VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in values.items()],
        )


def metadata(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as db:
        return {k: json.loads(v) for k, v in db.execute("SELECT * FROM metadata")}


@dataclass(frozen=True, slots=True)
class StoredEntry:
    id: str
    kind: Kind
    object_type: str
    name: str
    data: dict[str, Any]
    description: str
    categories: list[str]
    building_types: list[str]
    standards: list[str]
    climate_zones: list[str]
    roles: list[str]


def entries(path: Path, ids: Iterable[str] | None = None) -> Iterator[StoredEntry]:
    """Entries of the library, all or those with the given ids."""
    query = "SELECT * FROM reference_object"
    params: list[str] = []
    if ids is not None:
        params = list(ids)
        query += f" WHERE id IN ({','.join('?' * len(params))})"
    with sqlite3.connect(path) as db:
        for row in db.execute(query, params):
            (key, kind, object_type, name, data, description, *lists, _count) = row
            yield StoredEntry(
                key,
                kind,
                object_type,
                name,
                json.loads(data),
                description,
                *(json.loads(v) for v in lists),
            )


def embeddings(path: Path) -> Iterator[tuple[str, list[float]]]:
    with sqlite3.connect(path) as db:
        for key, blob in db.execute("SELECT id, vector FROM embedding"):
            yield key, unpack(blob)
