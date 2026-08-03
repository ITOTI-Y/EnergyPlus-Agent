from idfpy.models import IDFBaseModel
from idfpy.models.simulation import Building
from pydantic import BaseModel

from src._share import UNSET
from src.converters.base_converter import BaseConverter
from src.validator.data_model import LightSchema, PeopleSchema


class BuildingInputSchema(BaseModel):
    name: str
    axis: float
    source_only: str
    terrain: str = ""


class NullableInputSchema(BaseModel):
    value: str | None = "default"


class NullableTarget(IDFBaseModel):
    value: str | None = "default"


def test_to_idf_model_filters_source_fields_and_applies_overrides() -> None:
    source = BuildingInputSchema(
        name="Test Building",
        axis=15.0,
        source_only="ignored",
    )

    building = BaseConverter._to_idf_model(
        Building,
        source,
        north_axis=source.axis,
    )
    building_default = BaseConverter._to_idf_model(
        Building,
        source,
        terrain=UNSET,
    )

    assert isinstance(building, Building)
    assert building.name == "Test Building"
    assert building.north_axis == 15.0
    assert building.terrain == ""
    assert building_default.terrain == "Suburbs"


def test_to_idf_model_preserves_values_and_omits_only_unset() -> None:
    explicit_empty = BuildingInputSchema(
        name="Empty Terrain",
        axis=0.0,
        source_only="ignored",
        terrain="",
    )
    explicit_none = NullableInputSchema(value=None)

    building = BaseConverter._to_idf_model(Building, explicit_empty)
    nullable = BaseConverter._to_idf_model(NullableTarget, explicit_none)
    unset = BaseConverter._to_idf_model(
        NullableTarget,
        explicit_none,
        value=UNSET,
    )

    assert building.terrain == ""
    assert nullable.value is None
    assert unset.value == "default"


def test_optional_idf_fields_use_none_for_explicit_empty_values() -> None:
    light = LightSchema(
        name="Office Lights",
        zone_or_zone_list_or_space_or_space_list_name="Office",
        schedule_name="Always On",
        lighting_level=100.0,
        return_air_heat_gain_node_name="",
        exhaust_air_heat_gain_node_name="",
    )
    people = PeopleSchema(
        name="Office People",
        zone_or_zonelist_or_space_or_spacelist_name="Office",
        number_of_people_schedule_name="Always On",
        activity_level_schedule_name="Activity",
        number_of_people=2.0,
        work_efficiency_schedule_name="",
        thermal_comfort_model_1_type="",
    )

    assert light.return_air_heat_gain_node_name is None
    assert light.exhaust_air_heat_gain_node_name is None
    assert people.surface_name_angle_factor_list_name is None
    assert people.work_efficiency_schedule_name is None
    assert people.clothing_insulation_calculation_method_schedule_name is None
    assert people.clothing_insulation_schedule_name is None
    assert people.air_velocity_schedule_name is None
    assert people.thermal_comfort_model_1_type is None
    assert people.thermal_comfort_model_2_type is None
    assert people.thermal_comfort_model_3_type is None
    assert people.thermal_comfort_model_4_type is None
    assert people.thermal_comfort_model_5_type is None
    assert people.thermal_comfort_model_6_type is None
    assert people.thermal_comfort_model_7_type is None
    assert people.ankle_level_air_velocity_schedule_name is None
