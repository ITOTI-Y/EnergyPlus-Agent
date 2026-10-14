from pathlib import Path

import pytest
from idfpy import IDF
from idfpy.models.location import SizingPeriodDesignDay
from idfpy.models.simulation import SimulationControl, Timestep, Version

from src.state.defaults import add_design_days, new_model

SHENZHEN_DDY = Path(__file__).parents[2] / "data" / "weather" / "Shenzhen.ddy"


def test_new_model_has_global_objects():
    idf = new_model()

    for object_type in (Version, SimulationControl, Timestep):
        assert len(idf.all_of_type(object_type)) == 1


def test_add_design_days_picks_annual_heating_and_cooling():
    idf = new_model()

    add_design_days(idf, SHENZHEN_DDY)

    design_days = idf.all_of_type(SizingPeriodDesignDay)
    assert set(design_days) == {
        "Shenzhen Ann Htg 99.6% Condns DB",
        "Shenzhen Ann Clg .4% Condns DB=>MWB",
    }
    assert {d.day_type for d in design_days.values()} == {
        "WinterDesignDay",
        "SummerDesignDay",
    }


def test_add_design_days_requires_both_conditions(tmp_path):
    source = IDF.load(SHENZHEN_DDY)
    for name in list(source.all_of_type(SizingPeriodDesignDay)):
        if " Clg " in name:
            source.remove(SizingPeriodDesignDay, name)
    ddy = tmp_path / "heating_only.ddy"
    source.save(ddy)

    with pytest.raises(LookupError, match="Clg"):
        add_design_days(new_model(), ddy)
