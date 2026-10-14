import shutil
from pathlib import Path

import pytest
from idfpy import IDF
from idfpy.models import OutputVariable, SizingPeriodDesignDay

from src.mcp.tools.workflow import WorkflowTool
from src.runner.runner import SimulationResult
from src.state.config_state import ConfigState

DATA_DIR = Path(__file__).parents[3] / "data"
SHENZHEN_EPW = DATA_DIR / "weather" / "Shenzhen.epw"


@pytest.fixture
def written_models(monkeypatch) -> list[IDF]:
    """Replace EnergyPlus with a stub that keeps each IDF it is given."""
    models: list[IDF] = []

    def fake_run(idf_path: Path, _epw_path: Path, output_dir: Path) -> SimulationResult:
        models.append(IDF.load(idf_path))
        return SimulationResult(output_dir, 0, ())

    monkeypatch.setattr("src.mcp.tools.workflow.run_energyplus", fake_run)
    return models


def test_run_simulation_adds_default_output_variables(tmp_path, written_models):
    response = WorkflowTool(ConfigState()).run_simulation(
        str(SHENZHEN_EPW), str(tmp_path)
    )

    assert response.success, response.message
    assert len(written_models[0].all_of_type(OutputVariable)) == 7


def test_run_simulation_leaves_the_edited_model_unchanged(tmp_path, written_models):
    state = ConfigState()

    response = WorkflowTool(state).run_simulation(str(SHENZHEN_EPW), str(tmp_path))

    assert response.success, response.message
    assert len(written_models[0].all_of_type(SizingPeriodDesignDay)) == 2
    assert not state.idf.all_of_type(SizingPeriodDesignDay)
    assert not state.idf.all_of_type(OutputVariable)


def test_run_simulation_keeps_existing_output_variables(tmp_path, written_models):
    state = ConfigState()
    state.idf.add(
        OutputVariable(
            key_value="*",
            variable_name="Site Outdoor Air Drybulb Temperature",
            reporting_frequency="Timestep",
        )
    )

    response = WorkflowTool(state).run_simulation(str(SHENZHEN_EPW), str(tmp_path))

    assert response.success, response.message
    variables = list(written_models[0].all_of_type(OutputVariable).values())
    assert [v.variable_name for v in variables] == [
        "Site Outdoor Air Drybulb Temperature"
    ]


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
    assert isinstance(response.data, dict)
    written = IDF.load(Path(response.data["idf_path"]))
    assert len(written.all_of_type(SizingPeriodDesignDay)) == 2
    assert not state.idf.all_of_type(SizingPeriodDesignDay)
    assert (Path(response.data["output_dir"]) / "eplusout.eso").stat().st_size > 0
