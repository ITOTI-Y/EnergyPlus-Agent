import json
from typing import Literal

from idfpy.models import (
    BuildingSurfaceDetailed,
    Construction,
    FenestrationSurfaceDetailed,
)
from langchain_core.tools import BaseTool, tool

from src.mcp.geometry import surface_normal, surface_vertices
from src.mcp.state import ConfigState

# Aliases preserve the historical private names used throughout this module
# (call sites below unchanged). The implementations now live in the neutral
# ``src.mcp.geometry`` module so they can be shared with ``src.mcp.state``
# without a circular import (this module already imports ``src.mcp.state``).
_surface_normal = surface_normal
_wall_vertices = surface_vertices


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


# Fenestration surface_types that MUST be backed by a glazing (window)
# construction — i.e. one whose layer list contains a WindowMaterial. A
# plain Door is the only opaque-allowed type; Window / GlassDoor /
# TubularDaylight* all need glass. EnergyPlus aborts with a Severe error
# ("has an opaque surface construction; it should have a window
# construction") if this is violated, so we catch it at tool-call time
# and back-hop to material to create the missing WindowMaterial.
_GLAZING_SURFACE_TYPES = frozenset(
    {"Window", "GlassDoor", "TubularDaylightDiffuser", "TubularDaylightDome"}
)

# Construction layer fields, mirroring construction_tools._LAYER_FIELDS.
_CONSTRUCTION_LAYER_FIELDS = (
    "outside_layer",
    "layer_2",
    "layer_3",
    "layer_4",
    "layer_5",
    "layer_6",
    "layer_7",
    "layer_8",
    "layer_9",
    "layer_10",
)


def _is_glazing_construction(idf, construction_name: str) -> bool:
    """True if *construction_name* has at least one WindowMaterial layer.

    EnergyPlus classifies a construction as a "window construction" when any
    of its layers is a WindowMaterial:* object. The agent creates TWO
    window-material variants, both of which legally qualify a construction
    as glazing:

    - WindowMaterial:SimpleGlazingSystem — a simplified whole-window model
      (U/SHGC/VT only), created via create_glazing_material. Must be the
      sole layer.
    - WindowMaterial:Glazing — a true per-pane glass layer (thickness +
      per-pane optical data), created via create_glazing_layer_material.
      Composed with Material:AirGap in multi-pane assemblies.

    Checking only SimpleGlazingSystem (the original implementation) wrongly
    rejects a valid multi-pane window built from WindowMaterial:Glazing
    layers as "opaque", which silently drops every fenestration — the
    window is never created because create_fenestration refuses an "opaque"
    construction.
    """
    const = idf.get("Construction", construction_name)
    if const is None:
        return False
    for field in _CONSTRUCTION_LAYER_FIELDS:
        layer = getattr(const, field, None)
        if not layer:
            continue
        if idf.has("WindowMaterial:SimpleGlazingSystem", layer) or idf.has(
            "WindowMaterial:Glazing", layer
        ):
            return True
    return False


def _is_airboundary_construction(idf, construction_name: str) -> bool:
    """True if *construction_name* is a Construction:AirBoundary (open-air)."""
    return idf.has("Construction:AirBoundary", construction_name)


def _construction_exists(idf, name: str) -> bool:
    """True if *name* is a layered Construction or a Construction:AirBoundary."""
    return idf.has("Construction", name) or idf.has("Construction:AirBoundary", name)


def _parent_boundary_condition(idf, building_surface_name: str) -> str | None:
    """Outside boundary condition of a fenestration's parent base surface.

    Returns None if the parent surface does not exist. When the value is
    'Surface', the parent is a zone-separating (interzone) wall, which
    triggers EnergyPlus' rule that subsurfaces must also name an adjacent
    surface — the AirBoundary workaround bypasses that requirement.
    """
    parent = idf.get("BuildingSurface:Detailed", building_surface_name)
    if parent is None:
        return None
    return getattr(parent, "outside_boundary_condition", None)


def _parent_is_interzone(idf, building_surface_name: str) -> bool:
    """True if the parent base surface separates two zones (Surface boundary)."""
    return _parent_boundary_condition(idf, building_surface_name) == "Surface"


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
        return (
            None,
            "Parent surface has fewer than 3 vertices; cannot verify window orientation.",
        )

    win_pts = [(float(v["X"]), float(v["Y"]), float(v["Z"])) for v in vertices]
    wall_n = _surface_normal(wall_pts)
    if wall_n == (0.0, 0.0, 0.0):
        return (
            None,
            "Parent surface has degenerate (zero-area) geometry; cannot verify window orientation.",
        )

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
            f"(max vertex-to-plane distance={max_dist:.4f}m, limit={_COPLANARITY_TOLERANCE:.4f}m). Every window "
            "vertex must share the wall's plane coordinate."
        )

    # Check 2: normal direction (flip once if backward, then re-verify).
    win_n = _surface_normal(win_pts)
    if win_n == (0.0, 0.0, 0.0):
        return (
            None,
            "Window has degenerate (zero-area) geometry — vertices are collinear or duplicated.",
        )

    dot = win_n[0] * wall_n[0] + win_n[1] * wall_n[1] + win_n[2] * wall_n[2]
    if dot < 0:
        win_pts.reverse()  # single allowed flip; CCW <-> CW
        # Recompute normal after reversal (sign flips for a coplanar polygon).
        win_n = _surface_normal(win_pts)
        dot = win_n[0] * wall_n[0] + win_n[1] * wall_n[1] + win_n[2] * wall_n[2]

    if dot <= _NORMAL_DOT_TOLERANCE:
        return None, (
            "Window normal is still not consistent with the parent surface "
            f"after winding correction (dot={dot:.3f}). Re-check the window geometry."
        )

    aligned = [{"X": x, "Y": y, "Z": z} for (x, y, z) in win_pts]
    return aligned, None


def make_fenestration_tools(config: ConfigState) -> list[BaseTool]:
    idf = config._idf

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
        if idf is None:
            raise ValueError("IDF is None")
        if idf.has(FenestrationSurfaceDetailed, name):
            return _err(f"Fenestration '{name}' already exists.")
        if not _construction_exists(idf, construction_name):
            return _err(
                f"Construction '{construction_name}' not found. Create it in the construction phase first.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        # A window/glass-door/skylight MUST reference a glazing construction
        # (one whose layers include a WindowMaterial). An opaque construction
        # makes EnergyPlus abort with "has an opaque surface construction;
        # it should have a window construction". Back-hop to material so it
        # creates the WindowMaterial:SimpleGlazingSystem, after which
        # construction can rebuild the window construction with it.
        # EXCEPTION: an AirBoundary construction is the legitimate choice for
        # INTERIOR windows/glass-doors (open-air passage between zones); it
        # has no WindowMaterial and must NOT be flagged here. The dedicated
        # interzone check below governs AirBoundary usage.
        if (
            surface_type in _GLAZING_SURFACE_TYPES
            and not _is_glazing_construction(idf, construction_name)
            and not _is_airboundary_construction(idf, construction_name)
        ):
            return _err(
                f"Construction '{construction_name}' is opaque (no WindowMaterial "
                f"layer) and cannot be used for {surface_type} '{name}'. "
                f"Create a WindowMaterial:SimpleGlazingSystem glazing material "
                f"and rebuild the construction to use it.",
                {
                    "missing_ref": "WindowMaterial:SimpleGlazingSystem",
                    "missing_name": construction_name,
                },
            )
        if not idf.has("BuildingSurface:Detailed", building_surface_name):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {
                    "missing_ref": "BuildingSurface:Detailed",
                    "missing_name": building_surface_name,
                },
            )
        # Interior-subsurface rule: when the parent base surface separates two
        # zones (Outside Boundary Condition = 'Surface'), a subsurface that is
        # NOT an AirBoundary construction leaves its own Outside Boundary
        # Condition Object blank, which EnergyPlus rejects with "invalid blank
        # Outside Boundary Condition Object". The agent does not pair adjacent-
        # zone subsurfaces, so interior doors/windows MUST use a
        # Construction:AirBoundary (open-air) construction. Back-hop to
        # construction so it creates one.
        if _parent_is_interzone(
            idf, building_surface_name
        ) and not _is_airboundary_construction(idf, construction_name):
            return _err(
                f"Parent surface '{building_surface_name}' is an interior "
                f"(zone-separating) wall. {surface_type} '{name}' must use a "
                f"Construction:AirBoundary (create via "
                f"create_airboundary_construction) so it models an open "
                f"passage; '{construction_name}' is a regular layered "
                f"construction and would leave its boundary object blank.",
                {
                    "missing_ref": "Construction",
                    "missing_name": "Construction:AirBoundary",
                },
            )
        try:
            # Auto-correct window winding to match the parent wall's outward
            # normal. Reject (without storing) if the window cannot be made
            # consistent — i.e. it is not coplanar with the wall.
            wall_obj = idf.get("BuildingSurface:Detailed", building_surface_name)
            aligned, align_err = _align_window_to_wall(vertices, wall_obj)
            if align_err is not None or aligned is None:
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
            idf.add(FenestrationSurfaceDetailed(**kwargs))
            data = idf.get(FenestrationSurfaceDetailed, name)
            if data is None:
                raise ValueError("FenestrationSurface:Detailed not found")
            return _ok(
                f"Fenestration '{name}' created successfully.",
                data.model_dump(),
            )
        except Exception as e:
            return _err(f"Error creating fenestration '{name}': {e}")

    @tool
    def list_fenestrations() -> str:
        """List all fenestration surfaces."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [
            f.model_dump()
            for f in idf.all_of_type(FenestrationSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} fenestrations.", items)

    @tool
    def get_fenestration(name: str) -> str:
        """Read a fenestration by name."""
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(FenestrationSurfaceDetailed, name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        return _ok(f"Fenestration '{name}' read successfully.", obj.model_dump())

    @tool
    def update_fenestration(
        name: str,
        construction_name: str | None = None,
        building_surface_name: str | None = None,
        surface_type: Literal[
            "Door",
            "GlassDoor",
            "TubularDaylightDiffuser",
            "TubularDaylightDome",
            "Window",
        ]
        | None = None,
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
        if idf is None:
            raise ValueError("IDF is None")
        obj = idf.get(FenestrationSurfaceDetailed, name)
        if obj is None:
            return _err(f"Fenestration '{name}' not found.")
        if construction_name is not None and not _construction_exists(
            idf, construction_name
        ):
            return _err(
                f"Construction '{construction_name}' not found.",
                {"missing_ref": "Construction", "missing_name": construction_name},
            )
        # Glazing-construction check (mirrors create_fenestration). Uses the
        # EFFECTIVE surface_type (new value if provided, else the object's
        # existing one) and the EFFECTIVE construction (new if provided, else
        # the object's existing one) so reparenting a window onto a new
        # construction is validated the same way as a fresh create. An
        # AirBoundary construction is exempt (it is the valid choice for an
        # interior window/glass-door — see create_fenestration for rationale).
        if construction_name is not None:
            effective_type = surface_type or obj.surface_type
            effective_const = construction_name
        else:
            effective_type = surface_type or obj.surface_type
            effective_const = obj.construction_name
        if (
            effective_type in _GLAZING_SURFACE_TYPES
            and effective_const
            and not _is_glazing_construction(idf, effective_const)
            and not _is_airboundary_construction(idf, effective_const)
        ):
            return _err(
                f"Construction '{effective_const}' is opaque (no WindowMaterial "
                f"layer) and cannot be used for {effective_type} '{name}'. "
                f"Create a WindowMaterial:SimpleGlazingSystem glazing material "
                f"and rebuild the construction to use it.",
                {
                    "missing_ref": "WindowMaterial:SimpleGlazingSystem",
                    "missing_name": effective_const,
                },
            )
        if building_surface_name is not None and not idf.has(
            "BuildingSurface:Detailed", building_surface_name
        ):
            return _err(
                f"Parent surface '{building_surface_name}' not found.",
                {
                    "missing_ref": "BuildingSurface:Detailed",
                    "missing_name": building_surface_name,
                },
            )
        # Interior-subsurface rule (mirrors create_fenestration). Uses the
        # EFFECTIVE parent (new if provided, else current) and EFFECTIVE
        # construction so reparenting / re-assigning is validated like a fresh
        # create. An interior (Surface-boundary) parent requires the subsurface
        # to use a Construction:AirBoundary; a regular layered construction
        # would leave the boundary object blank and abort EnergyPlus.
        effective_parent = building_surface_name or obj.building_surface_name
        effective_const_for_interzone = construction_name or obj.construction_name
        if (
            _parent_is_interzone(idf, effective_parent)
            and effective_const_for_interzone
            and not _is_airboundary_construction(idf, effective_const_for_interzone)
        ):
            return _err(
                f"Parent surface '{effective_parent}' is an interior "
                f"(zone-separating) wall. '{name}' must use a "
                f"Construction:AirBoundary (create via "
                f"create_airboundary_construction); '{effective_const_for_interzone}' "
                f"is a regular layered construction.",
                {
                    "missing_ref": "Construction",
                    "missing_name": "Construction:AirBoundary",
                },
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
                        {
                            "missing_ref": "BuildingSurface:Detailed",
                            "missing_name": parent_name,
                        },
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
            return _ok(f"Fenestration '{name}' updated successfully.", obj.model_dump())
        except Exception as e:
            return _err(f"Error updating fenestration '{name}': {e}")

    @tool
    def delete_fenestration(name: str) -> str:
        """Delete a fenestration."""
        if idf is None:
            raise ValueError("IDF is None")
        if not idf.has(FenestrationSurfaceDetailed, name):
            return _err(f"Fenestration '{name}' not found.")
        idf.remove(FenestrationSurfaceDetailed, name)
        return _ok(f"Fenestration '{name}' deleted successfully.")

    @tool
    def list_surfaces() -> str:
        """Read-only: list parent surfaces a fenestration can attach to."""
        if idf is None:
            raise ValueError("IDF is None")
        items = [
            s.model_dump() for s in idf.all_of_type(BuildingSurfaceDetailed).values()
        ]
        return _ok(f"Listed {len(items)} surfaces.", items)

    @tool
    def list_constructions() -> str:
        """Read-only: list constructions a fenestration can reference."""
        if idf is None:
            raise ValueError("IDF is None")
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
