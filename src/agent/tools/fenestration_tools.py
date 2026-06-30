import json
import math

from idfpy.models.constructions import Construction
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.state import ConfigState


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


# Tolerance for the post-flip re-check in _align_window_to_wall. A backward
# window (dot < 0) is FIXED by the single allowed flip above, not rejected
# here — a coplanar window's dot sign simply inverts under reversal, so
# flipping once always turns a backward coplanar window positive. This check
# only catches the rare case where, after flipping, the dot is still at or
# below zero: geometry that slipped past the earlier coplanarity guard
# (Check 1) and zero-area guard. It is a last-resort safety net, not the
# primary defense.
_NORMAL_DOT_TOLERANCE = 0.01

# Maximum allowed distance (m) from any window vertex to the parent wall's
# plane. A window that passes the normal-direction check but has vertices
# off the wall plane is a warped/non-coplanar polygon — Newell's normal can
# still align with the wall by chance, so we additionally require every
# vertex to lie (near) on the plane. 1 cm tolerates float rounding.
_COPLANARITY_TOLERANCE = 0.01


def _surface_normal(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Unit outward normal of a polygon via Newell's method.

    Robust to non-strictly-planar quads (sums contributions per edge), so it
    matches EnergyPlus' own surface-normal computation.
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


def _max_distance_to_plane(
    points: list[tuple[float, float, float]],
    plane_normal: tuple[float, float, float],
    plane_point: tuple[float, float, float],
) -> float:
    """Largest absolute distance from any point to the plane."""
    a, b, c = plane_normal
    px, py, pz = plane_point
    worst = 0.0
    for x, y, z in points:
        d = abs((x - px) * a + (y - py) * b + (z - pz) * c)
        if d > worst:
            worst = d
    return worst


def _wall_vertices(wall_obj) -> list[tuple[float, float, float]]:
    """Extract (x, y, z) tuples from a BuildingSurface:Detailed object.

    idfpy stores wall geometry as a nested ``vertices`` list (each item has
    ``vertex_{x,y,z}_coordinate``), which is a different shape from
    FenestrationSurface's flat ``vertex_{i}_{x,y,z}_coordinate`` attributes.
    Read the nested list first, fall back to the flat attributes defensively.
    """
    nested = getattr(wall_obj, "vertices", None)
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
            val = getattr(wall_obj, f"vertex_{i}_{axis}_coordinate", None)
            if val is None:
                val = getattr(wall_obj, f"vertex_{axis}_coordinate_{i}", None)
            if val is not None:
                pts.setdefault(i, {})[axis] = float(val)
    return [(pts[i]["x"], pts[i]["y"], pts[i]["z"]) for i in sorted(pts)]


def _align_window_to_wall(
    vertices: list[dict[str, float]], wall_obj
) -> tuple[list[dict[str, float]] | None, str | None]:
    """Ensure the window lies on the parent wall and faces the same way.

    Two independent checks, because normal direction alone cannot tell a
    warped polygon from a flat one (Newell's normal can still align with
    the wall by chance for a non-coplanar quad):

      1. Coplanarity — every window vertex must lie within
         ``_COPLANARITY_TOLERANCE`` of the wall plane. Rejects windows that
         are tilted off the wall or not flat.
      2. Normal direction — ``dot(window_normal, wall_normal)`` must be
         positive after at most one winding flip. dot < 0 reverses the
         vertex order (the single allowed flip); a coplanar window is always
         fixed by that one flip. If the re-check is still at/below tolerance
         the geometry is degenerate (collinear/duplicate → zero-area normal).

    Returns (aligned_vertices, None) on success, or (None, error_message)
    when the window cannot be made consistent — callers must NOT store it.
    """
    wall_pts = _wall_vertices(wall_obj)
    if len(wall_pts) < 3:
        return None, "Parent surface has fewer than 3 vertices; cannot verify window orientation."

    win_pts = [(float(v["X"]), float(v["Y"]), float(v["Z"])) for v in vertices]
    wall_n = _surface_normal(wall_pts)
    if wall_n == (0.0, 0.0, 0.0):
        return None, "Parent surface has degenerate (zero-area) geometry; cannot verify window orientation."

    # Plane anchor = centroid of the wall vertices. Newell's normal is the
    # normal of the best-fit plane, and that plane passes through the
    # centroid, so the centroid is a more accurate on-plane reference point
    # than any single vertex (which may sit off the best-fit plane for a
    # slightly non-planar wall polygon).
    m = len(wall_pts)
    anchor = (
        sum(p[0] for p in wall_pts) / m,
        sum(p[1] for p in wall_pts) / m,
        sum(p[2] for p in wall_pts) / m,
    )

    # Check 1: coplanarity (vertex-to-plane distance).
    max_dist = _max_distance_to_plane(win_pts, wall_n, anchor)
    if max_dist > _COPLANARITY_TOLERANCE:
        return None, (
            "Window is not coplanar with the parent surface "
            "(max vertex-to-plane distance=%.4fm, limit=%.4fm). Every window "
            "vertex must share the wall's plane coordinate." % (max_dist, _COPLANARITY_TOLERANCE)
        )

    # Check 2: normal direction (flip once if backward, then re-verify).
    win_n = _surface_normal(win_pts)
    if win_n == (0.0, 0.0, 0.0):
        return None, "Window has degenerate (zero-area) geometry — vertices are collinear or duplicated."

    dot = win_n[0] * wall_n[0] + win_n[1] * wall_n[1] + win_n[2] * wall_n[2]
    if dot < 0:
        win_pts.reverse()  # single allowed flip; CCW <-> CW
        # Recompute normal after reversal (sign flips for a coplanar polygon).
        win_n = _surface_normal(win_pts)
        dot = win_n[0] * wall_n[0] + win_n[1] * wall_n[1] + win_n[2] * wall_n[2]

    if dot <= _NORMAL_DOT_TOLERANCE:
        return None, (
            "Window normal is still not consistent with the parent surface "
            "after winding correction (dot=%.3f). Re-check the window geometry." % dot
        )

    aligned = [{"X": x, "Y": y, "Z": z} for (x, y, z) in win_pts]
    return aligned, None


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @tool
    def create_fenestration(
        name: str,
        surface_type: str,
        construction_name: str,
        building_surface_name: str,
        vertices: list[dict[str, float]],
        multiplier: int = 1,
    ) -> str:
        """Create a FenestrationSurface:Detailed (window/door/skylight).

        Args:
            name: Unique fenestration name.
            surface_type: Window / Door / GlassDoor.
            construction_name: Existing Glazing construction name.
            building_surface_name: Existing parent Surface name.
            vertices: List of vertex dicts in meters. Each vertex is
                      `{"X": float, "Y": float, "Z": float}`. >= 3 points,
                      and MUST be coplanar with the parent surface (share the
                      wall's plane coordinate). Winding direction does not
                      matter — the tool auto-aligns the window's outward
                      normal to the parent wall's.
                      Example 1.5x1.2m window centered on a south wall at
                      sill 0.8m (wall at y=0, spans x=0..5):
                        [{"X": 1.75, "Y": 0.0, "Z": 0.8},
                         {"X": 3.25, "Y": 0.0, "Z": 0.8},
                         {"X": 3.25, "Y": 0.0, "Z": 2.0},
                         {"X": 1.75, "Y": 0.0, "Z": 2.0}]
            multiplier: Number of identical copies (>= 1).
        """
        if idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' already exists.")
        if not idf.has("Construction", construction_name):
            return _err(
                f"Construction '{construction_name}' not found. Create it in the construction phase first.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if not idf.has("BuildingSurface:Detailed", building_surface_name):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {"missing_ref": "BuildingSurface:Detailed", "missing_name": building_surface_name},
            )
        try:
            # Auto-correct window winding to match the parent wall's outward
            # normal. Reject (without storing) if the window cannot be made
            # consistent — i.e. it is not coplanar with the wall.
            wall_obj = idf.get("BuildingSurface:Detailed", building_surface_name)
            aligned, align_err = _align_window_to_wall(vertices, wall_obj)
            if align_err is not None:
                return _err(
                    f"Cannot create fenestration '{name}': {align_err}",
                    {"building_surface_name": building_surface_name},
                )
            vertices = aligned
            kwargs: dict = {
                "name": name,
                "surface_type": surface_type,
                "construction_name": construction_name,
                "building_surface_name": building_surface_name,
                "multiplier": float(multiplier),
                "number_of_vertices": len(vertices),
            }
            for i, v in enumerate(vertices, start=1):
                kwargs[f"vertex_{i}_x_coordinate"] = float(v["X"])
                kwargs[f"vertex_{i}_y_coordinate"] = float(v["Y"])
                kwargs[f"vertex_{i}_z_coordinate"] = float(v["Z"])
            fenestration = FenestrationSurfaceDetailed(**kwargs)
            idf.add(fenestration)
            return _ok(
                f"Fenestration '{name}' created successfully.",
                fenestration.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating fenestration '{name}': {e}")

    @tool
    def list_fenestrations() -> str:
        """List all fenestration surfaces."""
        items = [
            f.model_dump()
            for f in idf.all_of_type(FenestrationSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} fenestrations.", items)

    @tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        obj = idf.get(FenestrationSurfaceDetailed, name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        return _ok(f"Fenestration '{name}' read successfully.", obj.model_dump())

    @tool
    def update_fenestration(
        name: str,
        construction_name: str | None = None,
        building_surface_name: str | None = None,
        surface_type: str | None = None,
        multiplier: int | None = None,
        vertices: list[dict[str, float]] | None = None,
    ) -> str:
        """Update fields of an existing fenestration by name.

        Only non-None fields are written. To change geometry, pass a full
        new ``vertices`` list (replaces all existing vertices).

        Args:
            name: Existing fenestration name.
            construction_name: New glazing Construction name (must exist).
            building_surface_name: New parent Surface name (must exist).
            surface_type: Window / Door / GlassDoor.
            multiplier: Number of identical copies (>= 1).
            vertices: New full vertex list ({"X","Y","Z"} dicts), 3-4 points,
                      coplanar with the parent surface. Winding direction is
                      auto-aligned to the parent wall's outward normal.

        Note:
            Reparenting (passing ``building_surface_name`` without
            ``vertices``) moves the window to the new wall but does NOT
            re-derive its geometry — the existing vertices are kept as-is,
            which may no longer be coplanar with the new parent. To move a
            window to a different wall correctly, pass the new parent AND a
            fresh ``vertices`` list on that wall's plane.
        """
        obj = idf.get("FenestrationSurface:Detailed", name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        if construction_name is not None and not idf.has("Construction", construction_name):
            return _err(
                f"Construction '{construction_name}' not found.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        if building_surface_name is not None and not idf.has("BuildingSurface:Detailed", building_surface_name):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {"missing_ref": "BuildingSurface:Detailed", "missing_name": building_surface_name},
            )
        try:
            # Resolve the effective parent (new if provided, else current) and
            # auto-correct the new window winding BEFORE mutating any field,
            # so a rejected geometry leaves the existing object untouched.
            aligned_vertices: list[dict[str, float]] | None = None
            if vertices is not None:
                if len(vertices) < 3 or len(vertices) > 4:
                    return _err("Fenestration needs 3 or 4 vertices.")
                parent_name = building_surface_name or obj.building_surface_name
                wall_obj = idf.get("BuildingSurface:Detailed", parent_name)
                if wall_obj is None:
                    return _err(
                        f"Parent surface '{parent_name}' not found.",
                        {"missing_ref": "BuildingSurface:Detailed", "missing_name": parent_name},
                    )
                aligned, align_err = _align_window_to_wall(vertices, wall_obj)
                if align_err is not None:
                    return _err(
                        f"Cannot update fenestration '{name}': {align_err}",
                        {"building_surface_name": parent_name},
                    )
                aligned_vertices = aligned

            if construction_name is not None:
                obj.construction_name = construction_name
            if building_surface_name is not None:
                obj.building_surface_name = building_surface_name
            if surface_type is not None:
                obj.surface_type = surface_type
            if multiplier is not None:
                obj.multiplier = float(multiplier)
            if aligned_vertices is not None:
                vertices = aligned_vertices
                # Overwrite in place; do NOT blank fields first — vertex_1..3
                # are required `float` fields and reject None, so a
                # clear-then-set sequence raises ValidationError before the
                # new values are ever written (see thermal_zones.py:656-664).
                # vertex_4 is the only vertex that allows None.
                for i, v in enumerate(vertices, start=1):
                    setattr(obj, f"vertex_{i}_x_coordinate", float(v["X"]))
                    setattr(obj, f"vertex_{i}_y_coordinate", float(v["Y"]))
                    setattr(obj, f"vertex_{i}_z_coordinate", float(v["Z"]))
                if len(vertices) < 4:
                    for axis in ("x", "y", "z"):
                        setattr(obj, f"vertex_4_{axis}_coordinate", None)
                obj.number_of_vertices = len(vertices)
            return _ok(f"Fenestration '{name}' updated successfully.",
                       obj.model_dump())
        except Exception as e:
            return _err(f"Error updating fenestration '{name}': {e}")

    @tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        if not idf.has("FenestrationSurface:Detailed", name):
            return _err(f"Fenestration '{name}' not found.")
        idf.remove("FenestrationSurface:Detailed", name)
        return _ok(f"Fenestration '{name}' deleted successfully.")

    @tool
    def list_surfaces() -> str:
        """Read-only: list parent surfaces a fenestration can attach to."""
        items = [
            s.model_dump() for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a fenestration can reference."""
        items = [c.model_dump() for c in idf.all_of_type(Construction).values()]
        return _ok(f"Listed {len(items)} constructions.", items)

    return [
        create_fenestration,
        list_fenestrations,
        get_fenestration,
        update_fenestration,
        delete_fenestration,
        list_surfaces,
        list_constructions,
    ]
