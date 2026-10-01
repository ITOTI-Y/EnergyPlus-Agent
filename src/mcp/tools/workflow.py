import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.mcp.interface import ToolResponse
from src.mcp.state import ConfigState
from src.runner.runner import run_energyplus
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowTool:
    """High-level MCP workflow operations backed directly by idfpy."""

    def __init__(self, state: ConfigState):
        self.state = state

    def export_yaml(self, output_path: str) -> ToolResponse:
        try:
            path = Path(output_path)
            self.state.export_yaml(path)
            return ToolResponse(
                success=True,
                message=f"Exported YAML-like IDF snapshot to {path}",
                data={"path": str(path.absolute())},
            )
        except Exception as e:
            logger.exception("Error exporting YAML")
            return ToolResponse(success=False, message=f"Error exporting YAML: {e!s}")

    def export_idf(self, output_path: str = "./output/idf/output.idf") -> ToolResponse:
        try:
            path = self.state.save_idf(output_path)
            return ToolResponse(
                success=True,
                message=f"Exported IDF to {path}",
                data={"path": str(path.absolute())},
            )
        except Exception as e:
            logger.exception("Error exporting IDF")
            return ToolResponse(success=False, message=f"Error exporting IDF: {e!s}")

    def load_yaml(self, yaml_path: str) -> ToolResponse:
        try:
            path = Path(yaml_path)
            staged = ConfigState.load_yaml(path)
            self.state.update_from(staged)
            self.state.load_yaml_into_idf(path)
            summary = self.state.get_summary()
            return ToolResponse(
                success=True,
                message=f"Loaded YAML directly into IDF from {path}",
                data={"summary": summary.model_dump()},
            )
        except Exception as e:
            logger.exception("Error loading YAML")
            return ToolResponse(success=False, message=f"Error loading YAML: {e!s}")

    def validate_config(self) -> ToolResponse:
        errors = self.state.validate_references()
        if errors:
            return ToolResponse(
                success=False,
                message=f"Validation failed: {len(errors)} reference errors found.",
                data={"errors": errors},
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

        Validates references, writes the IDF into a fresh run directory under
        ``output_dir`` and runs EnergyPlus there.

        Args:
            epw_path: Path to the EPW weather data file.
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

        run_dir = (
            Path(output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True)
        idf_path = self.state.save_idf(run_dir / "in.idf")
        try:
            result = run_energyplus(idf_path, Path(epw_path), run_dir)
        except (FileNotFoundError, TimeoutError) as e:
            logger.exception("EnergyPlus run failed")
            return ToolResponse(success=False, message=f"EnergyPlus run failed: {e!s}")

        data = {
            "idf_path": str(idf_path),
            "output_dir": str(result.output_dir),
            "return_code": result.return_code,
            "errors": [asdict(message) for message in result.errors],
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
