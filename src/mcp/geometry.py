"""Neutral geometry helpers shared across the codebase.

Lives in ``src/mcp`` so it can be imported by both ``src/mcp/state.py`` and
``src/agent/tools/fenestration_tools.py`` without creating a circular import
(``fenestration_tools`` already imports from ``src.mcp.state``).

Two concerns are provided here:

* ``surface_normal`` — unit outward normal of a planar polygon via Newell's
  method. Robust to non-strictly-planar quads (it sums per-edge
  contributions), which matches EnergyPlus' own surface-normal computation.
* ``surface_vertices`` — read the (x, y, z) tuples from an idfpy
  ``BuildingSurface:Detailed`` object, handling both the nested ``vertices``
  list shape and the flat ``vertex_{i}_{axis}_coordinate`` fallback.
"""

from __future__ import annotations

import math

# Tolerance for a normal-direction dot product comparison. A surface whose
# normal is essentially perpendicular to the expected axis (dot within
# +/- this of zero) is treated as "sideways" and left untouched — only
# clearly-wrong normals (e.g. a Floor pointing firmly up) are flipped.
NORMAL_DOT_TOLERANCE = 0.01

# Maximum allowed distance (m) from any vertex to a reference plane. Used by
# callers (e.g. fenestration coplanarity check) to detect warped polygons;
# 1 cm tolerates float rounding.
COPLANARITY_TOLERANCE = 0.01


def surface_normal(
    vertices: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Unit outward normal of a polygon via Newell's method.

    Robust to non-strictly-planar quads (sums contributions per edge), so it
    matches EnergyPlus' own surface-normal computation. Returns ``(0, 0, 0)``
    for degenerate (collinear / duplicate) input.
    """
    nx = ny = nz = 0.0
    n = len(vertices)
    for i in range(n):
        ax, ay, az = vertices[i]
        bx, by, bz = vertices[(i + 1) % n]
        nx += (ay - by) * (az + bz)
        ny += (az - bz) * (ax + bx)
        nz += (ax - bx) * (ay + by)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        return (0.0, 0.0, 0.0)  # degenerate (collinear / duplicate) vertices
    return (nx / length, ny / length, nz / length)


def surface_vertices(obj) -> list[tuple[float, float, float]]:
    """Extract ``(x, y, z)`` tuples from an idfpy surface object.

    Handles two storage shapes:

    * ``BuildingSurface:Detailed`` stores geometry as a nested ``vertices``
      list (each item has ``vertex_{x,y,z}_coordinate``).
    * Some legacy/flat shapes store ``vertex_{i}_{x,y,z}_coordinate`` or
      ``vertex_{x,y,z}_coordinate_{i}`` as direct attributes.

    The nested list is tried first; the flat attributes are a defensive
    fallback. Returns at least 3 points when available.
    """
    nested = getattr(obj, "vertices", None)
    if nested:
        pts = []
        for item in nested:
            x = getattr(item, "vertex_x_coordinate", None)
            y = getattr(item, "vertex_y_coordinate", None)
            z = getattr(item, "vertex_z_coordinate", None)
            if x is not None and y is not None and z is not None:
                pts.append((float(x), float(y), float(z)))
        if len(pts) >= 3:
            return pts

    # Fallback: flat vertex_{i}_{axis}_coordinate attributes.
    pts: dict[int, dict[str, float]] = {}
    for i in range(1, 5):
        for axis in ("x", "y", "z"):
            val = getattr(obj, f"vertex_{i}_{axis}_coordinate", None)
            if val is None:
                val = getattr(obj, f"vertex_{axis}_coordinate_{i}", None)
            if val is not None:
                pts.setdefault(i, {})[axis] = float(val)
    return [(pts[i]["x"], pts[i]["y"], pts[i]["z"]) for i in sorted(pts)]
