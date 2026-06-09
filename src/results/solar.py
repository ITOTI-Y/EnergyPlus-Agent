"""Surface solar irradiation helpers for 3-D visualization.

Reads annual-mean incident solar from ``eplusout.eso`` when available;
otherwise estimates relative exposure from exterior surface orientation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from src.results.idf_geometry import SurfacePolygon, ZoneGeometry, _strip_comments, _split_objects, _parse_fields

_SOLAR_VAR = "Surface Outside Face Incident Solar Radiation Rate per Area"
_COMMENT_RE = re.compile(r"!-[^\n]*")


def _surface_normal(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Unit normal from the first three vertices (Newell-style for quads)."""
    if len(vertices) < 3:
        return (0.0, 0.0, 1.0)
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def iter_exterior_surfaces(zones: dict[str, ZoneGeometry]) -> list[SurfacePolygon]:
    """Return outdoor-exposed surfaces (walls and roofs facing Outdoors)."""
    out: list[SurfacePolygon] = []
    for zg in zones.values():
        for s in zg.surfaces:
            bc = s.outside_boundary.strip().lower()
            st = s.surface_type.strip().lower()
            if bc == "outdoors" and st in ("wall", "roof"):
                out.append(s)
    return out


def parse_idf_latitude(idf_path: Path) -> float:
    """Read latitude from Site:Location (degrees). Default Shenzhen ~22.55."""
    text = _strip_comments(idf_path.read_text(encoding="utf-8", errors="replace"))
    for block in _split_objects(text):
        fields = _parse_fields(block)
        if fields and fields[0].lower() == "site:location" and len(fields) > 2:
            try:
                return float(fields[2])
            except ValueError:
                pass
    return 22.55


def estimate_surface_solar_proxy(
    surfaces: list[SurfacePolygon],
    latitude_deg: float = 22.55,
) -> dict[str, float]:
    """Orientation-based relative solar exposure index (0–1) per surface name.

    Not physical W/m² — used when simulation did not output surface solar.
    """
    lat = math.radians(latitude_deg)
    # South azimuth in building coordinates (+Y)
    south = (0.0, 1.0, 0.0)
    result: dict[str, float] = {}
    for surf in surfaces:
        nx, ny, nz = _surface_normal(surf.vertices)
        # Outward normal should point away from zone; use abs alignment with sky
        cos_tilt = abs(nz)  # 0=vertical wall, 1=horizontal
        horiz = math.sqrt(nx * nx + ny * ny) or 1e-9
        # Horizontal component alignment with south
        south_factor = max(0.0, (ny / horiz) * (1.0 - abs(nz)) + abs(nz) * max(0.0, math.sin(lat)))
        sky_view = (1.0 + cos_tilt) / 2.0
        index = south_factor * sky_view
        result[surf.name] = round(max(0.05, min(1.0, index)), 4)
    return result


def parse_eso_surface_solar_mean(eso_path: Path) -> dict[str, float]:
    """Parse ``eplusout.eso`` and return annual mean W/m² per surface name.

    Returns an empty dict if the solar variable was not requested in the run.
    """
    if not eso_path.exists():
        return {}

    lines = eso_path.read_text(encoding="utf-8", errors="replace").splitlines()
    report_ids: dict[int, str] = {}  # id -> surface name
    in_dict = True
    for ln in lines:
        if ln.strip() == "End of Data Dictionary":
            in_dict = False
            continue
        if in_dict:
            if _SOLAR_VAR not in ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 3:
                try:
                    rid = int(parts[0])
                    sname = parts[2].strip()
                    report_ids[rid] = sname
                except ValueError:
                    continue
        else:
            break

    if not report_ids:
        return {}

    sums: dict[str, float] = {n: 0.0 for n in report_ids.values()}
    counts: dict[str, int] = {n: 0 for n in report_ids.values()}

    for ln in lines:
        if in_dict or ln.startswith("End") or ln.startswith("Program"):
            continue
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        try:
            rid = int(parts[0])
        except ValueError:
            continue
        if rid not in report_ids:
            continue
        try:
            val = float(parts[1])
        except ValueError:
            continue
        sname = report_ids[rid]
        sums[sname] += val
        counts[sname] += 1

    return {
        name: round(sums[name] / counts[name], 2)
        for name in sums
        if counts[name] > 0
    }


def resolve_surface_solar(
    zones: dict[str, ZoneGeometry],
    run_dir: Path,
    idf_path: Path | None,
) -> tuple[dict[str, float], str, str]:
    """Return (surface_name -> W/m² or index), unit label, data source note."""
    exterior = iter_exterior_surfaces(zones)
    eso_path = run_dir / "eplusout.eso"
    measured = parse_eso_surface_solar_mean(eso_path)
    if measured:
        # Fill missing exterior surfaces with 0
        full = {s.name: measured.get(s.name, 0.0) for s in exterior}
        return full, "W/m²", "simulation (ESO)"

    lat = parse_idf_latitude(idf_path) if idf_path else 22.55
    proxy = estimate_surface_solar_proxy(exterior, lat)
    return proxy, "relative index", "orientation estimate (re-run with surface solar output for W/m²)"
