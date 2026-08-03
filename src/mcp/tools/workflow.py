import time
from pathlib import Path

from src.mcp.interface import ToolResponse
from src.mcp.state import ConfigState
from src.runner.runner import EnergyPlusRunner
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

        Validates references, exports to YAML, converts to IDF, and
        executes the EnergyPlus simulation.

        Args:
            epw_path: Path to the EPW weather data file.
            output_dir: Directory for simulation output files.

        Returns:
            ToolResponse with IDF path and output directory on success.
        """
        try:
            validation = self.validate_config()
            if not validation.success:
                return ToolResponse(
                    success=False,
                    message="Validation reference errors, cannot run simulation.",
                    data=validation.data,
                )

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_idf = Path(output_dir) / f"temp_{timestamp}.idf"
            self.state.save_idf(temp_idf)

            runner = EnergyPlusRunner()
            ok = runner.run_idf(
                epw_path,
                idf_file_path=temp_idf,
                output_directory=Path(output_dir),
            )
            if not ok:
                return ToolResponse(
                    success=False,
                    message="EnergyPlus simulation failed.",
                    data={
                        "idf_path": str(temp_idf.absolute()),
                        "output_dir": output_dir,
                    },
                )

            return ToolResponse(
                success=True,
                message="Simulation run successfully.",
                data={"idf_path": str(temp_idf.absolute()), "output_dir": output_dir},
            )
        except Exception as e:
            logger.exception("Error running simulation")
            return ToolResponse(
                success=False, message=f"Error running simulation: {e!s}"
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
