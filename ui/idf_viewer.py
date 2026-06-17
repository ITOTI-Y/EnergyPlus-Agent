"""IDF → interactive 3-D building model (Plotly Mesh3d).

Public API
----------
build_idf_3d_model(idf_path) -> plotly.graph_objects.Figure

Renders every ``BuildingSurface:Detailed`` (walls / floors / roofs / ceilings)
as a solid mesh, and cuts real holes into walls for every
``FenestrationSurface:Detailed`` (windows / doors) using a trimesh boolean
difference.  A semi-transparent glazing patch is layered over each opening so
the viewer shows both the aperture and the glass.

Robustness is the priority: if the boolean backend (manifold3d / blender) is
unavailable, or a particular wall fails to boolean-subtract (non-convex,
degenerate geometry, coordinate drift), that single wall silently falls back
to an un-cut solid face.  The figure always renders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh

from src.results.idf_geometry import (
    FenestrationPolygon,
    SurfacePolygon,
    ZoneGeometry,
    parse_fenestrations,
    parse_idf_geometry,
)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

_WALL_EXTRUDE = 0.04          # wall solid thickness along its outward normal (m)
_WINDOW_EXTRUDE = 0.12        # window cutter thickness — must exceed wall (m)
_BOOLEAN_TIMEOUT_ATTEMPTS = 1 # try boolean once per wall; fallback on any failure

# Base colours per surface class.  Window colour is applied to the glazing
# patch; wall colour is applied to the (possibly hole-punched) wall solid.
_TYPE_COLORS: dict[str, str] = {
    "wall": "#b0b6bd",     # light grey
    "floor": "#a0784f",    # warm brown
    "roof": "#5a6168",     # dark grey
    "ceiling": "#9fb4c4",  # pale blue
}
_WINDOW_COLOR = "rgba(80, 160, 230, 0.45)"  # semi-transparent glazing blue


# ---------------------------------------------------------------------------
# Boolean backend probe (run once, cached)
# ---------------------------------------------------------------------------

_boolean_available: bool | None = None


def _probe_boolean() -> bool:
    """Return True if trimesh boolean difference is usable in this env.

    Cached after the first call.  When False, every wall skips the boolean
    path and renders as a solid un-cut face.
    """
    global _boolean_available
    if _boolean_available is not None:
        return _boolean_available
    try:
        from trimesh.boolean import difference  # noqa: F401

        a = trimesh.creation.box([2.0, 2.0, 0.1])
        b = trimesh.creation.box([0.5, 0.5, 0.3])
        res = difference([a, b])
        _boolean_available = bool(res is not None and len(res.faces) > 0)
    except Exception:
        _boolean_available = False
    return _boolean_available


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _surface_normal(
    vertices: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Unit normal of a planar polygon via the first three vertices (Newell-ish)."""
    if len(vertices) < 3:
        return (0.0, 0.0, 1.0)
    (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = vertices[0], vertices[1], vertices[2]
    ax, ay, az = x1 - x0, y1 - y0, z1 - z0
    bx, by, bz = x2 - x0, y2 - y0, z2 - z0
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _project_to_plane(
    point: tuple[float, float, float],
    plane_point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Project *point* onto the plane through *plane_point* with *normal*."""
    nx, ny, nz = normal
    dx = point[0] - plane_point[0]
    dy = point[1] - plane_point[1]
    dz = point[2] - plane_point[2]
    d = dx * nx + dy * ny + dz * nz
    return (point[0] - d * nx, point[1] - d * ny, point[2] - d * nz)


def _make_prism(
    polygon_verts: list[tuple[float, float, float]],
    normal: tuple[float, float, float],
    depth: float,
) -> trimesh.Trimesh | None:
    """Build a watertight prism by extruding *polygon_verts* along *normal*.

    Does not depend on shapely.  Returns None for degenerate input
    (fewer than 3 vertices).
    """
    n = len(polygon_verts)
    if n < 3:
        return None
    base = np.asarray(polygon_verts, dtype=float)
    offset = np.asarray(normal, dtype=float) * depth
    verts = np.vstack([base, base + offset])

    faces: list[list[int]] = []
    # Bottom face — fan, reversed winding so the normal points down at z=base
    for i in range(1, n - 1):
        faces.append([0, i + 1, i])
    # Top face — fan, normal points along +offset
    for i in range(1, n - 1):
        faces.append([n, n + i, n + i + 1])
    # Side quads (two triangles each)
    for i in range(n):
        a = i
        b = (i + 1) % n
        faces.append([a, b, b + n])
        faces.append([a, b + n, a + n])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh


# ---------------------------------------------------------------------------
# Aggregation container
# ---------------------------------------------------------------------------


@dataclass
class _MeshBucket:
    """Accumulates Plotly Mesh3d arrays for one surface class (one trace)."""

    xs: list[float]
    ys: list[float]
    zs: list[float]
    i: list[int]
    j: list[int]
    k: list[int]
    names: list[str]
    base: int = 0  # vertex index offset for the next appended mesh

    def append_mesh(self, mesh: trimesh.Trimesh, name: str) -> None:
        v = mesh.vertices
        f = mesh.faces
        self.xs.extend(v[:, 0].tolist())
        self.ys.extend(v[:, 1].tolist())
        self.zs.extend(v[:, 2].tolist())
        self.i.extend((f[:, 0] + self.base).tolist())
        self.j.extend((f[:, 1] + self.base).tolist())
        self.k.extend((f[:, 2] + self.base).tolist())
        # One hover label per triangle, all pointing at the same surface name
        self.names.extend([name] * len(f))
        self.base += len(v)


def _new_bucket() -> _MeshBucket:
    return _MeshBucket(xs=[], ys=[], zs=[], i=[], j=[], k=[], names=[])


# ---------------------------------------------------------------------------
# Core: render one wall with holes punched for its fenestrations
# ---------------------------------------------------------------------------


def _build_wall_mesh(
    wall: SurfacePolygon,
    windows: list[FenestrationPolygon],
) -> trimesh.Trimesh | None:
    """Return a wall Trimesh with windows cut out, or None to signal
    'use the un-cut fallback'.  Never raises.
    """
    if not _probe_boolean():
        return None

    normal = _surface_normal(wall.vertices)
    wall_solid = _make_prism(wall.vertices, normal, _WALL_EXTRUDE)
    if wall_solid is None:
        return None

    # Build one cutter per window, projecting window verts onto the wall plane
    # to absorb coordinate drift between FenestrationSurface and its host wall.
    plane_point = wall.vertices[0]
    cutters: list[trimesh.Trimesh] = []
    for win in windows:
        if len(win.vertices) < 3:
            continue
        snapped = [
            _project_to_plane(v, plane_point, normal) for v in win.vertices
        ]
        cutter = _make_prism(snapped, normal, _WINDOW_EXTRUDE)
        if cutter is not None:
            cutters.append(cutter)

    if not cutters:
        return wall_solid  # no windows on this wall — return the solid prism

    try:
        from trimesh.boolean import difference

        result = difference([wall_solid, *cutters])
        if result is None or len(result.faces) == 0:
            return None
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level figure builder
# ---------------------------------------------------------------------------


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        title=message,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_idf_3d_model(idf_path: Path) -> go.Figure:
    """Render an IDF file as an interactive 3-D building model.

    Walls are solid meshes with real holes cut for every window/door
    (trimesh boolean difference).  A semi-transparent glazing patch is
    layered over each opening.  Floors / roofs / ceilings are coloured solids.

    Parameters
    ----------
    idf_path:
        Path to an EnergyPlus ``.idf`` file.

    Returns
    -------
    plotly.graph_objects.Figure
        Always returns a figure — malformed geometry degrades gracefully to
        an un-cut solid wall rather than raising.
    """
    idf_path = Path(idf_path)
    if not idf_path.exists():
        return _empty_figure(f"IDF file not found: {idf_path}")

    try:
        zones: dict[str, ZoneGeometry] = parse_idf_geometry(idf_path)
        fenestrations: list[FenestrationPolygon] = parse_fenestrations(idf_path)
    except Exception as exc:
        return _empty_figure(f"Failed to parse IDF: {exc}")

    if not zones:
        return _empty_figure("No BuildingSurface:Detailed objects found in IDF.")

    # Index fenestrations by their host surface name for fast lookup
    windows_by_host: dict[str, list[FenestrationPolygon]] = {}
    for fen in fenestrations:
        windows_by_host.setdefault(fen.building_surface_name, []).append(fen)

    buckets: dict[str, _MeshBucket] = {
        "wall": _new_bucket(),
        "floor": _new_bucket(),
        "roof": _new_bucket(),
        "ceiling": _new_bucket(),
    }
    win_bucket = _new_bucket()
    boolean_active = _probe_boolean()
    cut_count = 0
    fallback_count = 0

    for zone in zones.values():
        for surface in zone.surfaces:
            stype = surface.surface_type.strip().lower()
            bucket_key = stype if stype in buckets else "wall"

            if bucket_key == "wall":
                host_windows = windows_by_host.get(surface.name, [])
                if host_windows and boolean_active:
                    mesh = _build_wall_mesh(surface, host_windows)
                    if mesh is not None:
                        buckets["wall"].append_mesh(mesh, surface.name)
                        cut_count += 1
                    else:
                        # Fallback: render the wall as an un-cut solid prism
                        fallback = _make_prism(
                            surface.vertices,
                            _surface_normal(surface.vertices),
                            _WALL_EXTRUDE,
                        )
                        if fallback is not None:
                            buckets["wall"].append_mesh(fallback, surface.name)
                        fallback_count += 1
                else:
                    fallback = _make_prism(
                        surface.vertices,
                        _surface_normal(surface.vertices),
                        _WALL_EXTRUDE,
                    )
                    if fallback is not None:
                        buckets["wall"].append_mesh(fallback, surface.name)
            else:
                # Floor / roof / ceiling: thin solid prism
                prism = _make_prism(
                    surface.vertices,
                    _surface_normal(surface.vertices),
                    _WALL_EXTRUDE,
                )
                if prism is not None:
                    buckets[bucket_key].append_mesh(prism, surface.name)

    # Glazing patches: every fenestration rendered as a thin semi-transparent
    # panel sitting on its wall plane.  Independent of whether the wall was
    # cut, so the viewer always shows glass in the opening.
    for fen in fenestrations:
        if len(fen.vertices) < 3:
            continue
        normal = _surface_normal(fen.vertices)
        # Nudge glazing to the middle of the wall thickness so it reads as
        # sitting inside the aperture rather than on one face.
        nudge = np.asarray(normal) * (_WALL_EXTRUDE * 0.5)
        nudged = [
            (v[0] + nudge[0], v[1] + nudge[1], v[2] + nudge[2])
            for v in fen.vertices
        ]
        panel = _make_prism(nudged, normal, 0.005)
        if panel is not None:
            win_bucket.append_mesh(panel, fen.name)

    # ------------------------------------------------------------------ #
    # Assemble figure
    # ------------------------------------------------------------------ #
    fig = go.Figure()

    def _add_bucket_trace(bucket: _MeshBucket, name: str, color: str) -> None:
        if not bucket.xs:
            return
        fig.add_trace(
            go.Mesh3d(
                x=bucket.xs,
                y=bucket.ys,
                z=bucket.zs,
                i=bucket.i,
                j=bucket.j,
                k=bucket.k,
                color=color,
                opacity=1.0,
                name=name,
                showlegend=False,
                hovertemplate=(
                    "<b>%{customdata}</b><extra></extra>"
                ),
                customdata=np.array(bucket.names).reshape(-1, 1),
                flatshading=True,
                lighting=dict(ambient=0.75, diffuse=0.8, specular=0.1),
            )
        )

    _add_bucket_trace(buckets["wall"], "Wall", _TYPE_COLORS["wall"])
    _add_bucket_trace(buckets["floor"], "Floor", _TYPE_COLORS["floor"])
    _add_bucket_trace(buckets["roof"], "Roof", _TYPE_COLORS["roof"])
    _add_bucket_trace(buckets["ceiling"], "Ceiling", _TYPE_COLORS["ceiling"])

    if win_bucket.xs:
        fig.add_trace(
            go.Mesh3d(
                x=win_bucket.xs,
                y=win_bucket.ys,
                z=win_bucket.zs,
                i=win_bucket.i,
                j=win_bucket.j,
                k=win_bucket.k,
                color=_WINDOW_COLOR,
                opacity=0.55,
                name="Window",
                showlegend=False,
                hovertemplate="<b>%{customdata}</b> (glazing)<extra></extra>",
                customdata=np.array(win_bucket.names).reshape(-1, 1),
            )
        )

    n_zones = len(zones)
    n_surfaces = sum(len(z.surfaces) for z in zones.values())
    title_parts = [
        f"Building 3D Model — {idf_path.name}",
        f"{n_zones} zones · {n_surfaces} surfaces · {len(fenestrations)} openings",
    ]
    if not boolean_active:
        title_parts.append("(boolean backend unavailable — windows shown as glazing only)")
    elif fallback_count:
        title_parts.append(f"({fallback_count} wall(s) fell back to un-cut)")
    elif cut_count:
        title_parts.append(f"({cut_count} wall(s) with real openings)")

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X (m)", backgroundcolor="rgb(245,245,250)"),
            yaxis=dict(title="Y (m)", backgroundcolor="rgb(245,245,250)"),
            zaxis=dict(title="Z (m)", backgroundcolor="rgb(245,245,250)"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.1),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        title="<br>".join(title_parts),
        margin=dict(l=0, r=0, t=90, b=0),
        height=680,
    )
    return fig
