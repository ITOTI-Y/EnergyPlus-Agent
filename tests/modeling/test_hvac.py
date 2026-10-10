import pytest
from idfpy import IDF
from idfpy.models.hvac_templates import HVACTemplateThermostat

from src.modeling.errors import ModelingError
from src.modeling.hvac import check_setpoints_free


def test_a_second_thermostat_for_the_same_setpoints_is_refused():
    idf = IDF()
    idf.add(
        HVACTemplateThermostat(
            name="Office",
            heating_setpoint_schedule_name="Heat",
            cooling_setpoint_schedule_name="Cool",
        )
    )

    with pytest.raises(ModelingError, match="give these zones 'Office'"):
        check_setpoints_free(idf, "Heat", "Cool")
    check_setpoints_free(idf, "Heat", "Cool_Lobby")
