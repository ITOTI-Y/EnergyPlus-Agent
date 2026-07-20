"""IDF geometry parser for EnergyPlus input files.

Extracts BuildingSurface:Detailed objects from an IDF file (pure text
parsing, no idfpy dependency) and organises them by zone into
ZoneGeometry objects ready for 3-D visualisation.

Public API
----------
parse_idf_geometry(idf_path) -> dict[str, ZoneGeometry]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SurfacePolygon:
    """One planar polygon surface from an IDF BuildingSurface:Detailed object."""

    name: str
    surface_type: str  # "Wall" | "Floor" | "Roof" | "Ceiling"
    zone_name: str
    outside_boundary: str  # "Outdoors" | "Surface" | "Ground" | "Adiabatic" …
    vertices: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class FenestrationPolygon:
    """One planar polygon opening (window/door/glass door) from an IDF
    FenestrationSurface:Detailed object.

    Lives on the plane of its host ``building_surface_name`` (a Wall usually).
    Vertices are already in the same world coordinate system as the host
    BuildingSurface:Detailed, so they can be rendered directly without any
    additional transform.
    """

    name: str
    surface_type: str  # "Window" | "Door" | "GlassDoor" | "TubularDaylightDome" | "TubularDaylightDiffuser"
    construction_name: str
    building_surface_name: str  # host wall/surface name
    vertices: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class ZoneGeometry:
    """All surfaces belonging to one thermal zone."""

    zone_name: str
    surfaces: list[SurfacePolygon] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived helpers (computed lazily via properties)
    # ------------------------------------------------------------------

    @property
    def floor_vertices(self) -> list[tuple[float, float, float]]:
        """Vertices of the first Floor surface found (x, y, z)."""
        for s in self.surfaces:
            if s.surface_type.lower() == "floor":
                return s.vertices
        return []

    @property
    def ceiling_z(self) -> float:
        """Maximum z-coordinate across all surfaces (proxy for ceiling height)."""
        zs = [v[2] for s in self.surfaces for v in s.vertices]
        return max(zs) if zs else 3.0

    @property
    def floor_z(self) -> float:
        """Minimum z-coordinate across all surfaces (floor level)."""
        zs = [v[2] for s in self.surfaces for v in s.vertices]
        return min(zs) if zs else 0.0

    @property
    def all_vertices(self) -> list[tuple[float, float, float]]:
        """Flat list of all unique vertices (deduplicated to 6 dp)."""
        seen: set[tuple] = set()
        out: list[tuple[float, float, float]] = []
        for s in self.surfaces:
            for v in s.vertices:
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def bounding_box(self) -> dict[str, float]:
        """Return min/max for each axis."""
        xs = [v[0] for s in self.surfaces for v in s.vertices]
        ys = [v[1] for s in self.surfaces for v in s.vertices]
        zs = [v[2] for s in self.surfaces for v in s.vertices]
        if not xs:
            return {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
        return {
            "xmin": min(xs),
            "xmax": max(xs),
            "ymin": min(ys),
            "ymax": max(ys),
            "zmin": min(zs),
            "zmax": max(zs),
        }

    def centroid(self) -> tuple[float, float, float]:
        """Geometric centroid of the bounding box."""
        bb = self.bounding_box()
        return (
            (bb["xmin"] + bb["xmax"]) / 2,
            (bb["ymin"] + bb["ymax"]) / 2,
            (bb["zmin"] + bb["zmax"]) / 2,
        )


# ---------------------------------------------------------------------------
# IDF text parser
# ---------------------------------------------------------------------------

# Strip inline comments: everything from "!-" to end of line
_COMMENT_RE = re.compile(r"!-[^\n]*")
# BuildingSurface:Detailed fields (positional, 0-indexed including object class):
#  0  BuildingSurface:Detailed  (class name)
#  1  name
#  2  surface_type
#  3  construction_name
#  4  zone_name
#  5  space_name
#  6  outside_boundary_condition
#  7  outside_boundary_condition_object
#  8  sun_exposure
#  9  wind_exposure
# 10  view_factor_to_ground
# 11  number_of_vertices
# 12+ x1, y1, z1, x2, y2, z2, ...
_VERTEX_FIELD_START = 12

# FenestrationSurface:Detailed fields (positional, 0-indexed including class):
#  0  FenestrationSurface:Detailed  (class name)
#  1  name
#  2  surface_type
#  3  construction_name
#  4  building_surface_name
#  5  outside_boundary_condition_object
#  6  view_factor_to_ground
#  7  frame_and_divider_name
#  8  multiplier
#  9  number_of_vertices
# 10+ x1, y1, z1, x2, y2, z2, ...
_FENESTRATION_VERTEX_FIELD_START = 10


def _strip_comments(text: str) -> str:
    return _COMMENT_RE.sub("", text)


def _split_objects(text: str) -> list[str]:
    """Split IDF text into individual object strings (split on ';')."""
    return [blk.strip() for blk in text.split(";") if blk.strip()]


def _parse_fields(block: str) -> list[str]:
    """Return comma-split field values from a single IDF object block.

    First element is the object class name (e.g. 'BuildingSurface:Detailed').
    """
    # Collapse whitespace and newlines, then split on comma
    collapsed = re.sub(r"\s+", " ", block).strip()
    return [f.strip() for f in collapsed.split(",")]


def _parse_vertices(fields: list[str], start: int) -> list[tuple[float, float, float]]:
    """Extract (x, y, z) triplets from field list starting at *start*."""
    vertices: list[tuple[float, float, float]] = []
    i = start
    while i + 2 < len(fields):
        try:
            x = float(fields[i])
            y = float(fields[i + 1])
            z = float(fields[i + 2])
            vertices.append((x, y, z))
        except ValueError:
            pass
        i += 3
    return vertices


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def parse_idf_geometry(idf_path: Path) -> dict[str, ZoneGeometry]:
    """Parse all BuildingSurface:Detailed objects from *idf_path*.

    Returns a dict mapping **original zone name** (e.g. ``"Zone_Core"``) to
    :class:`ZoneGeometry`.

    Parameters
    ----------
    idf_path:
        Path to an EnergyPlus ``.idf`` file.

    Returns
    -------
    dict[str, ZoneGeometry]
        Keys are zone names as written in the IDF (mixed case).
    """
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    text = _strip_comments(text)
    blocks = _split_objects(text)

    zones: dict[str, ZoneGeometry] = {}

    for block in blocks:
        fields = _parse_fields(block)
        if not fields:
            continue

        obj_type = fields[0].strip()

        if obj_type.lower() == "buildingsurface:detailed":
            if len(fields) < _VERTEX_FIELD_START + 3:
                continue  # malformed, skip

            name = fields[1]
            surface_type = fields[2]
            # construction_name = fields[3]  # not needed
            zone_name = fields[4]
            # space_name = fields[5]
            outside_bc = fields[6] if len(fields) > 6 else ""
            # fields[7] = outside_bc_object (may be empty)
            # fields[8] = sun_exposure
            # fields[9] = wind_exposure
            # fields[10] = view_factor_to_ground
            # fields[11] = number_of_vertices (Autocalculate or integer)
            # fields[12+] = x1, y1, z1, x2, y2, z2, ...

            vertices = _parse_vertices(fields, _VERTEX_FIELD_START)
            if not vertices:
                continue

            surface = SurfacePolygon(
                name=name,
                surface_type=surface_type,
                zone_name=zone_name,
                outside_boundary=outside_bc,
                vertices=vertices,
            )

            if zone_name not in zones:
                zones[zone_name] = ZoneGeometry(zone_name=zone_name)
            zones[zone_name].surfaces.append(surface)

    return zones


def parse_fenestrations(idf_path: Path) -> list[FenestrationPolygon]:
    """Parse all FenestrationSurface:Detailed objects from *idf_path*.

    Returns a flat list of :class:`FenestrationPolygon` (windows, doors,
    glass doors).  Vertices are in the same world coordinate system as the
    host BuildingSurface:Detailed, so they can be rendered directly.

    Parameters
    ----------
    idf_path:
        Path to an EnergyPlus ``.idf`` file.

    Returns
    -------
    list[FenestrationPolygon]
    """
    text = idf_path.read_text(encoding="utf-8", errors="replace")
    text = _strip_comments(text)
    blocks = _split_objects(text)

    fenestrations: list[FenestrationPolygon] = []

    for block in blocks:
        fields = _parse_fields(block)
        if not fields:
            continue

        if fields[0].strip().lower() != "fenestrationsurface:detailed":
            continue
        if len(fields) < _FENESTRATION_VERTEX_FIELD_START + 3:
            continue  # malformed, skip

        name = fields[1]
        surface_type = fields[2]
        construction_name = fields[3] if len(fields) > 3 else ""
        building_surface_name = fields[4] if len(fields) > 4 else ""

        vertices = _parse_vertices(fields, _FENESTRATION_VERTEX_FIELD_START)
        if not vertices:
            continue

        fenestrations.append(
            FenestrationPolygon(
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                building_surface_name=building_surface_name,
                vertices=vertices,
            )
        )

    return fenestrations


# ---------------------------------------------------------------------------
# Utility: zone-name normalisation (IDF ↔ CSV key)
# ---------------------------------------------------------------------------


def idf_zone_to_csv_key(zone_name: str) -> str:
    """Convert IDF zone name to CSV column key style.

    Example: ``"Zone_Core"`` → ``"ZONE_CORE"``
    """
    return zone_name.upper().replace(" ", "_")


def csv_key_to_idf_zone(csv_key: str, zones: dict[str, ZoneGeometry]) -> str | None:
    """Find the IDF zone name matching *csv_key* (case-insensitive)."""
    target = csv_key.upper().replace(" ", "_")
    for zone_name in zones:
        if zone_name.upper().replace(" ", "_") == target:
            return zone_name
    return None
