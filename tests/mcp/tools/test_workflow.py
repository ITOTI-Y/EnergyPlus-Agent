import shutil
from pathlib import Path

import pytest
from idfpy.models.location import SizingPeriodDesignDay

from src.mcp.tools.workflow import WorkflowTool
from src.state.config_state import ConfigState

DATA_DIR = Path(__file__).parents[3] / "data"


def test_run_simulation_requires_ddy_beside_epw(tmp_path):
    epw = tmp_path / "weather.epw"
    shutil.copy(DATA_DIR / "weather" / "Shenzhen.epw", epw)

    response = WorkflowTool(ConfigState()).run_simulation(str(epw), str(tmp_path))

    assert not response.success
    assert "weather.ddy" in response.message
    assert not list(tmp_path.glob("run_*"))


@pytest.mark.skipif(shutil.which("energyplus") is None, reason="EnergyPlus not on PATH")
def test_run_simulation_adds_design_days_and_succeeds(tmp_path):
    state = ConfigState()
    state.load_model(DATA_DIR / "schemas" / "building_schema.epJSON")

    response = WorkflowTool(state).run_simulation(
        str(DATA_DIR / "weather" / "Shenzhen.epw"), str(tmp_path)
    )

    assert response.success, response.data
    assert len(state.idf.all_of_type(SizingPeriodDesignDay)) == 2
