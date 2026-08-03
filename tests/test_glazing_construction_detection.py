"""Regression test: _is_glazing_construction must recognize BOTH window
material variants, not just SimpleGlazingSystem.

The original implementation only checked ``idf.has("WindowMaterial:
SimpleGlazingSystem", layer)``, which wrongly classified a valid multi-pane
window built from WindowMaterial:Glazing layers (created via
create_glazing_layer_material) as "opaque". create_fenestration then
refused to create the fenestration — silently dropping every window in the
model. This was the root cause of the 0-window result across all 10 office
test cases: material and construction phases built the glazing correctly,
but fenestration_tools rejected it.

EnergyPlus accepts two window-material types as qualifying a construction
as glazing:
  - WindowMaterial:SimpleGlazingSystem (whole-window U/SHGC/VT model)
  - WindowMaterial:Glazing (true per-pane glass layer)
"""

from idfpy.models.constructions import (
    Construction,
    Material,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.agent.tools.fenestration_tools import _is_glazing_construction
from src.mcp.state import ConfigState, _idf_values

_VERTS = [
    BuildingSurfaceDetailedVerticesItem(
        vertex_x_coordinate=0.0, vertex_y_coordinate=0.0, vertex_z_coordinate=0.0
    )
] * 3


def _state() -> ConfigState:
    return ConfigState()


def _make_glazing_layer(name="Clear_Glass_3mm"):
    """Build a valid WindowMaterial:Glazing per-pane layer."""
    return WindowMaterialGlazing(
        name=name,
        optical_data_type="SpectralAverage",
        thickness=0.003,
        solar_transmittance_at_normal_incidence=0.83,
        front_side_solar_reflectance_at_normal_incidence=0.07,
        back_side_solar_reflectance_at_normal_incidence=0.07,
        visible_transmittance_at_normal_incidence=0.90,
        front_side_visible_reflectance_at_normal_incidence=0.08,
        back_side_visible_reflectance_at_normal_incidence=0.08,
        infrared_transmittance_at_normal_incidence=0.0,
        front_side_infrared_hemispherical_emissivity=0.84,
        back_side_infrared_hemispherical_emissivity=0.84,
        conductivity=1.0,
    )


def test_simple_glazing_construction_is_detected():
    """A construction whose sole layer is WindowMaterial:SimpleGlazingSystem
    must be detected as glazing (the case the original code handled)."""
    s = _state()
    s.idf.add(WindowMaterialSimpleGlazingSystem(
        name="Win_Simple", u_factor=1.8,
        solar_heat_gain_coefficient=0.4, visible_transmittance=0.7,
    ))
    s.idf.add(Construction(name="Win_Const_Simple", outside_layer="Win_Simple"))

    assert _is_glazing_construction(s.idf, "Win_Const_Simple") is True


def test_per_pane_glazing_construction_is_detected():
    """A multi-pane construction whose layers include WindowMaterial:Glazing
    (created via create_glazing_layer_material) MUST also be detected as
    glazing. This is the regression case — the original code returned False
    here, rejecting every multi-pane window."""
    s = _state()
    s.idf.add(_make_glazing_layer("Clear_Glass_3mm"))
    s.idf.add(Material(
        name="Air_Gap_13mm", roughness="rough", thickness=0.013,
        conductivity=0.026, density=1.2, specific_heat=1005.0,
    ))
    # Double-pane window: glass + air gap + glass
    s.idf.add(Construction(
        name="Win_Double_Pane", outside_layer="Clear_Glass_3mm",
        layer_2="Air_Gap_13mm", layer_3="Clear_Glass_3mm",
    ))

    assert _is_glazing_construction(s.idf, "Win_Double_Pane") is True


def test_opaque_construction_is_not_detected_as_glazing():
    """A construction with only opaque Material layers must NOT be detected
    as glazing (the legitimate rejection case)."""
    s = _state()
    s.idf.add(Material(
        name="Brick", roughness="rough", thickness=0.1,
        conductivity=0.7, density=1400.0, specific_heat=840.0,
    ))
    s.idf.add(Construction(name="Wall_Const", outside_layer="Brick"))

    assert _is_glazing_construction(s.idf, "Wall_Const") is False


def test_nonexistent_construction_is_not_glazing():
    """A missing construction name must return False, not raise."""
    s = _state()
    assert _is_glazing_construction(s.idf, "Does_Not_Exist") is False


def test_create_fenestration_accepts_per_pane_glazing_construction():
    """End-to-end: create_fenestration must ACCEPT a window whose
    construction uses WindowMaterial:Glazing layers (the scenario that
    failed in production with "Construction 'X' is opaque"). This is the
    user-visible behavior the fix restores."""
    from src.agent.tools import make_fenestration_tools

    s = _state()
    s.idf.add(_make_glazing_layer("Glazing_Layer"))
    s.idf.add(Construction(name="Win_Const", outside_layer="Glazing_Layer"))
    s.idf.add(Zone(name="Z1"))
    s.idf.add(BuildingSurfaceDetailed(
        name="South_Wall", surface_type="Wall", construction_name="Win_Const",
        zone_name="Z1", outside_boundary_condition="Outdoors",
        sun_exposure="SunExposed", wind_exposure="WindExposed",
        vertices=[
            BuildingSurfaceDetailedVerticesItem(vertex_x_coordinate=0.0, vertex_y_coordinate=0.0, vertex_z_coordinate=0.0),
            BuildingSurfaceDetailedVerticesItem(vertex_x_coordinate=5.0, vertex_y_coordinate=0.0, vertex_z_coordinate=0.0),
            BuildingSurfaceDetailedVerticesItem(vertex_x_coordinate=5.0, vertex_y_coordinate=0.0, vertex_z_coordinate=3.0),
            BuildingSurfaceDetailedVerticesItem(vertex_x_coordinate=0.0, vertex_y_coordinate=0.0, vertex_z_coordinate=3.0),
        ],
    ))

    tools = make_fenestration_tools(s)
    create_fen = [t for t in tools if t.name == "create_fenestration"][0]
    result = create_fen.invoke({
        "name": "South_Window", "surface_type": "Window",
        "construction_name": "Win_Const", "building_surface_name": "South_Wall",
        "vertices": [
            {"X": 1.0, "Y": 0.0, "Z": 1.0},
            {"X": 4.0, "Y": 0.0, "Z": 1.0},
            {"X": 4.0, "Y": 0.0, "Z": 2.5},
            {"X": 1.0, "Y": 0.0, "Z": 2.5},
        ],
        "multiplier": 1,
    })

    import json
    payload = json.loads(result)
    assert payload["success"] is True, f"create_fenestration failed: {payload.get('message')}"
    # The window must actually be in the IDF now.
    assert len(_idf_values(s.idf, "FenestrationSurface:Detailed")) == 1
