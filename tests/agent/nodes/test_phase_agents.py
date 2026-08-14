"""Cassette-based integration tests for phase agent nodes.

LLM responses are recorded once (`pytest --record-mode=once` with a real
LLM_API_KEY) and replayed offline afterwards; tools execute for real on
every run, so the assertions cover prompt -> tool call -> IDF mutation.
"""

import pytest
from idfpy import IDF
from idfpy.models.constructions import (
    Construction,
    Material,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.hvac_templates import HVACTemplateZoneIdealLoadsAirSystem
from idfpy.models.internal_gains import Lights, People
from idfpy.models.schedules import (
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    FenestrationSurfaceDetailed,
    Zone,
)

from src.agent.nodes.construction import construction_agent
from src.agent.nodes.fenestration import fenestration_agent
from src.agent.nodes.hvac import hvac_agent
from src.agent.nodes.lights import lights_agent
from src.agent.nodes.material import material_agent
from src.agent.nodes.people import people_agent
from src.agent.nodes.schedule import schedule_agent
from src.agent.nodes.surface import surface_agent
from src.agent.state import AgentState
from src.mcp.state import ConfigState

pytestmark = pytest.mark.usefixtures("pinned_llm_env")

BRICK = Material(
    name="Brick_100mm",
    roughness="MediumRough",
    thickness=0.1,
    conductivity=0.89,
    density=1920.0,
    specific_heat=790.0,
)


def _constant_schedule(name: str, type_limits: str, value: float) -> ScheduleCompact:
    return ScheduleCompact(
        name=name,
        schedule_type_limits_name=type_limits,
        data=[
            ScheduleCompactDataItem(field="Through: 12/31"),
            ScheduleCompactDataItem(field="For: AllDays"),
            ScheduleCompactDataItem(field=f"Until: 24:00, {value}"),
        ],
    )


def _seed_zone(idf: IDF) -> None:
    idf.add(Zone(name="F1_Office"))


def _seed_wall(idf: IDF) -> None:
    """6m x 3m south wall of F1_Office at y=0, referencing ExtWall_Simple."""
    idf.add(BRICK.model_copy(deep=True))
    idf.add(Construction(name="ExtWall_Simple", outside_layer="Brick_100mm"))
    idf.add(
        BuildingSurfaceDetailed(
            name="F1_Office_South_Wall",
            surface_type="Wall",
            construction_name="ExtWall_Simple",
            zone_name="F1_Office",
            outside_boundary_condition="Outdoors",
            sun_exposure="SunExposed",
            wind_exposure="WindExposed",
            number_of_vertices=4,
            vertices=[
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=x,
                    vertex_y_coordinate=0.0,
                    vertex_z_coordinate=z,
                )
                for x, z in [(0.0, 3.0), (0.0, 0.0), (6.0, 0.0), (6.0, 3.0)]
            ],
        )
    )


def _seed_fraction_limits(idf: IDF) -> None:
    idf.add(
        ScheduleTypeLimits(
            name="Fraction",
            lower_limit_value=0.0,
            upper_limit_value=1.0,
            numeric_type="Continuous",
        )
    )


@pytest.mark.vcr
def test_material_agent_creates_material():
    state = AgentState(
        user_input=(
            "Create exactly one standard material named 'Brick_100mm': "
            "thickness 0.1 m, conductivity 0.89 W/m-K, density 1920 kg/m3, "
            "specific heat 790 J/kg-K, roughness MediumRough."
        )
    )

    out = material_agent(state)

    materials = out["config_state"].idf.all_of_type(Material)
    assert "Brick_100mm" in materials
    assert materials["Brick_100mm"].thickness == 0.1
    assert str(out["messages"][0].content).startswith("[material]")


@pytest.mark.vcr
def test_construction_agent_creates_construction():
    seeded = ConfigState()
    seeded.idf.add(BRICK.model_copy(deep=True))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one construction named 'ExtWall_Simple' with a "
            "single layer 'Brick_100mm'."
        ),
    )

    out = construction_agent(state)

    constructions = out["config_state"].idf.all_of_type(Construction)
    assert "ExtWall_Simple" in constructions
    assert constructions["ExtWall_Simple"].outside_layer == "Brick_100mm"
    assert str(out["messages"][0].content).startswith("[construction]")
    assert out["upstream_request"] == {}


@pytest.mark.vcr
def test_schedule_agent_creates_schedule():
    state = AgentState(
        user_input=(
            "Create the ScheduleTypeLimits 'Fraction' (0.0 to 1.0, CONTINUOUS, "
            "Dimensionless) and exactly one Schedule:Compact named "
            "'Office_Occupancy' using it: 1.0 for all days, all year."
        )
    )

    out = schedule_agent(state)

    schedules = out["config_state"].idf.all_of_type(ScheduleCompact)
    assert "Office_Occupancy" in schedules
    assert str(out["messages"][0].content).startswith("[schedule]")


@pytest.mark.vcr
def test_surface_agent_creates_surface():
    seeded = ConfigState()
    _seed_zone(seeded.idf)
    seeded.idf.add(BRICK.model_copy(deep=True))
    seeded.idf.add(Construction(name="ExtWall_Simple", outside_layer="Brick_100mm"))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one floor surface named 'F1_Office_Floor' for zone "
            "'F1_Office' with construction 'ExtWall_Simple': a 6m x 6m square "
            "at ground level (z=0, corners (0,0), (6,0), (6,6), (0,6)), "
            "outside boundary condition Ground."
        ),
    )

    out = surface_agent(state)

    surfaces = out["config_state"].idf.all_of_type(BuildingSurfaceDetailed)
    assert "F1_Office_Floor" in surfaces
    assert surfaces["F1_Office_Floor"].zone_name == "F1_Office"
    assert str(out["messages"][0].content).startswith("[surface]")
    assert out["upstream_request"] == {}


@pytest.mark.vcr
def test_fenestration_agent_creates_window():
    seeded = ConfigState()
    _seed_zone(seeded.idf)
    _seed_wall(seeded.idf)
    seeded.idf.add(
        WindowMaterialSimpleGlazingSystem(
            name="Glass_U18", u_factor=1.8, solar_heat_gain_coefficient=0.4
        )
    )
    seeded.idf.add(Construction(name="Window_Simple", outside_layer="Glass_U18"))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one window named 'F1_Office_South_Wall_Window' on "
            "surface 'F1_Office_South_Wall' with construction 'Window_Simple': "
            "1.5m wide x 1.2m tall, centered horizontally, sill at 0.8m."
        ),
    )

    out = fenestration_agent(state)

    fens = out["config_state"].idf.all_of_type(FenestrationSurfaceDetailed)
    assert "F1_Office_South_Wall_Window" in fens
    assert fens["F1_Office_South_Wall_Window"].building_surface_name == (
        "F1_Office_South_Wall"
    )
    assert str(out["messages"][0].content).startswith("[fenestration]")
    assert out["upstream_request"] == {}


@pytest.mark.vcr
def test_hvac_agent_creates_thermostat_and_ideal_loads():
    seeded = ConfigState()
    _seed_zone(seeded.idf)
    seeded.idf.add(
        ScheduleTypeLimits(
            name="Temperature",
            lower_limit_value=-100.0,
            upper_limit_value=100.0,
            numeric_type="Continuous",
        )
    )
    seeded.idf.add(_constant_schedule("Heating_Setpoint", "Temperature", 20.0))
    seeded.idf.add(_constant_schedule("Cooling_Setpoint", "Temperature", 24.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one thermostat named 'Office_Thermostat' using "
            "heating setpoint schedule 'Heating_Setpoint' and cooling setpoint "
            "schedule 'Cooling_Setpoint', then one IdealLoadsAirSystem for "
            "zone 'F1_Office' using that thermostat."
        ),
    )

    out = hvac_agent(state)

    idf = out["config_state"].idf
    assert "Office_Thermostat" in idf.all_of_type("HVACTemplate:Thermostat")
    ideal_loads = idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).values()
    assert "F1_Office" in {ils.zone_name for ils in ideal_loads}
    assert str(out["messages"][0].content).startswith("[hvac]")


@pytest.mark.vcr
def test_people_agent_creates_people():
    seeded = ConfigState()
    _seed_zone(seeded.idf)
    _seed_fraction_limits(seeded.idf)
    seeded.idf.add(
        ScheduleTypeLimits(
            name="Activity Level",
            lower_limit_value=0.0,
            upper_limit_value=1000.0,
            numeric_type="Continuous",
        )
    )
    seeded.idf.add(_constant_schedule("Office_Occupancy", "Fraction", 1.0))
    seeded.idf.add(_constant_schedule("Office_Activity", "Activity Level", 120.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one People object named 'F1_Office_People' for "
            "zone 'F1_Office': method People/Area with 0.1 people/m2, "
            "occupancy schedule 'Office_Occupancy', activity level schedule "
            "'Office_Activity'."
        ),
    )

    out = people_agent(state)

    people = out["config_state"].idf.all_of_type(People)
    assert "F1_Office_People" in people
    assert str(out["messages"][0].content).startswith("[people]")


@pytest.mark.vcr
def test_lights_agent_creates_lights():
    seeded = ConfigState()
    _seed_zone(seeded.idf)
    _seed_fraction_limits(seeded.idf)
    seeded.idf.add(_constant_schedule("Office_Lighting", "Fraction", 1.0))
    state = AgentState(
        config_state=seeded,
        user_input=(
            "Create exactly one Lights object named 'F1_Office_Lights' for "
            "zone 'F1_Office': method Watts/Area at 10 W/m2, schedule "
            "'Office_Lighting'."
        ),
    )

    out = lights_agent(state)

    lights = out["config_state"].idf.all_of_type(Lights)
    assert "F1_Office_Lights" in lights
    assert str(out["messages"][0].content).startswith("[lights]")
