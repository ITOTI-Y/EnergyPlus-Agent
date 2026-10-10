import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from idfpy.models.location import SizingPeriodDesignDay

from src.mcp.interface import ToolResponse
from src.modeling.ground import use_kiva_foundations
from src.modeling.validation import (
    completeness_issues,
    model_issues,
    simulation_issues,
)
from src.runner.runner import run_energyplus
from src.state.config_state import ConfigState
from src.state.defaults import add_design_days
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowTool:
    """High-level MCP workflow operations backed directly by idfpy."""

    def __init__(self, state: ConfigState):
        self.state = state

    def export_model(self, output_path: str) -> ToolResponse:
        try:
            path = self.state.save_model(Path(output_path))
        except (OSError, ValueError) as e:
            logger.exception("Error exporting model")
            return ToolResponse(success=False, message=f"Error exporting model: {e!s}")
        return ToolResponse(
            success=True,
            message=f"Exported model to {path}",
            data={"path": str(path.absolute())},
        )

    def load_model(self, input_path: str) -> ToolResponse:
        try:
            self.state.load_model(Path(input_path))
        except (OSError, ValueError) as e:
            logger.exception("Error loading model")
            return ToolResponse(success=False, message=f"Error loading model: {e!s}")
        return ToolResponse(
            success=True,
            message=f"Loaded model from {input_path}",
            data={"summary": self.state.get_summary().model_dump()},
        )

    def validate_config(self) -> ToolResponse:
        idf = self.state.idf
        issues = model_issues(idf) + completeness_issues(idf)
        if issues:
            return ToolResponse(
                success=False,
                message=f"Validation failed: {len(issues)} problem(s) found.",
                data={"errors": [asdict(issue) for issue in issues]},
            )
        return ToolResponse(
            success=True,
            message="Validation passed.",
            data=self.state.get_summary().model_dump(),
        )

    def run_simulation(
        self, epw_path: str, output_dir: str = "./output"
    ) -> ToolResponse:
        """Run an EnergyPlus simulation with the current configuration.

        Validates references, adds the annual design days from the ``.ddy``
        file next to the EPW when the model has none, moves ground-contact
        floors onto a Kiva foundation unless the model fixes its own ground
        temperature, writes the IDF into a fresh run directory under
        ``output_dir`` and runs EnergyPlus there.

        Args:
            epw_path: Path to the EPW weather data file; ``<stem>.ddy`` must
                sit beside it unless the model already has design days.
            output_dir: Parent directory for per-run output directories.

        Returns:
            ToolResponse with the IDF path, the run directory and the
            EnergyPlus Severe/Fatal messages.
        """
        validation = self.validate_config()
        if not validation.success:
            return ToolResponse(
                success=False,
                message="Validation reference errors, cannot run simulation.",
                data=validation.data,
            )

        epw = Path(epw_path)
        if not self.state.idf.all_of_type(SizingPeriodDesignDay):
            try:
                add_design_days(self.state.idf, epw.with_suffix(".ddy"))
            except (FileNotFoundError, LookupError) as e:
                return ToolResponse(
                    success=False, message=f"Cannot add design days: {e!s}"
                )

        if floors := use_kiva_foundations(self.state.idf):
            logger.info("Ground floors simulated with Kiva: {}", ", ".join(floors))

        run_dir = (
            Path(output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True)
        idf_path = self.state.save_model(run_dir / "in.idf")
        try:
            result = run_energyplus(idf_path, epw, run_dir)
        except (FileNotFoundError, TimeoutError) as e:
            logger.exception("EnergyPlus run failed")
            return ToolResponse(success=False, message=f"EnergyPlus run failed: {e!s}")

        data = {
            "idf_path": str(idf_path),
            "output_dir": str(result.output_dir),
            "return_code": result.return_code,
            "errors": [
                asdict(issue)
                for issue in simulation_issues(self.state.idf, result.messages)
            ],
        }
        if not result.succeeded:
            return ToolResponse(
                success=False,
                message=f"EnergyPlus reported {len(result.errors)} error message(s).",
                data=data,
            )
        return ToolResponse(
            success=True, message="Simulation run successfully.", data=data
        )

    def get_summary(self) -> ToolResponse:
        return ToolResponse(
            success=True,
            message="Configuration summary.",
            data=self.state.get_summary().model_dump(),
        )

    def clear_all(self) -> ToolResponse:
        self.state.clear()
        logger.info("All IDF configuration cleared.")
        return ToolResponse(success=True, message="All configuration cleared.")
