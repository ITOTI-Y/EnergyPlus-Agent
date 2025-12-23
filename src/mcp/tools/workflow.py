import time
from pathlib import Path

from src.converter_manager import ConverterManager
from src.mcp.interface import ToolResponse
from src.mcp.state import ConfigState
from src.runner.runner import EnergyPlusRunner
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowTool:
    def __init__(self, state: ConfigState):
        self.state = state

    def export_yaml(self, output_path: str) -> ToolResponse:
        try:
            path = Path(output_path)
            self.state.export_yaml(path)
            return ToolResponse(
                success=True,
                message=f"Exported YAML to {path}",
                data={"path": str(path.absolute())},
            )
        except Exception as e:
            logger.exception("Error exporting YAML")
            return ToolResponse(
                success=False,
                message=f"Error exporting YAML: {e!s}",
            )

    def load_yaml(self, yaml_path: str) -> ToolResponse:
        try:
            path = Path(yaml_path)
            new_state = ConfigState.load_yaml(path)

            self.state.clear()

            self.state.building = new_state.building
            self.state.site_location = new_state.site_location
            self.state.zones = new_state.zones
            self.state.materials = new_state.materials
            self.state.constructions = new_state.constructions
            self.state.surfaces = new_state.surfaces
            self.state.fenestrations = new_state.fenestrations
            self.state.schedules = new_state.schedules
            self.state.hvac = new_state.hvac
            self.state.simulation_control = new_state.simulation_control
            self.state.run_period = new_state.run_period
            self.state.global_geometry_rules = new_state.global_geometry_rules
            self.state.output_variable_dictionary = new_state.output_variable_dictionary
            self.state.output_diagnostics = new_state.output_diagnostics
            self.state.output_table_summary_reports = new_state.output_table_summary_reports
            self.state.output_variable = new_state.output_variable
            self.state.output_control_table_style = new_state.output_control_table_style

            summary = self.state.get_summary()
            return ToolResponse(
                success=True,
                message=f"Loaded YAML from {path}",
                data={"summary": summary.model_dump()},
            )

        except Exception as e:
            logger.exception("Error loading YAML")
            return ToolResponse(
                success=False,
                message=f"Error loading YAML: {e!s}",
            )

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
        try:
            validation = self.validate_config()
            if not validation.success:
                return ToolResponse(
                    success=False,
                    message="Validation Reference Errors, cannot run simulation.",
                    data=validation.data,
                )

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            temp_yaml = Path(output_dir) / f"temp_{timestamp}.yaml"
            temp_idf = Path(output_dir) / f"temp_{timestamp}.idf"

            self.state.export_yaml(temp_yaml)

            manager = ConverterManager(temp_yaml)
            manager.convert_all()
            manager.save_idf(temp_idf)

            runner = EnergyPlusRunner(idf=manager.idf)
            runner.run_idf(epw_path)

            logger.info(f"Simulation run successfully. Output directory: {output_dir}")

            return ToolResponse(
                success=True,
                message="Simulation run successfully.",
                data={"idf_path": str(temp_idf.absolute()), "output_dir": output_dir},
            )

        except Exception as e:
            logger.exception("Error running simulation")
            return ToolResponse(
                success=False,
                message=f"Error running simulation: {e!s}",
            )

    def get_summary(self) -> ToolResponse:
        return ToolResponse(
            success=True,
            message="Configuration summary.",
            data=self.state.get_summary().model_dump(),
        )

    def clear_all(self) -> ToolResponse:
        self.state.clear()
        logger.info("All configuration cleared.")
        return ToolResponse(
            success=True,
            message="All configuration cleared.",
        )
