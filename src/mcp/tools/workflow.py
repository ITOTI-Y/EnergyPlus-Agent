import time
from pathlib import Path

from src.mcp.interface import ToolResponse
from src.mcp.state import ConfigState, _idf_values
from src.runner.runner import EnergyPlusRunner
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_geometry_completeness(state: ConfigState) -> list[str]:
    """Return blocking geometry-completeness errors before simulation.

    Reference validation catches dangling names, but an empty or envelope-less
    model can still pass those checks and either run as an empty building or
    fail late in EnergyPlus. This preflight gate prevents those false
    successes and gives the revise loop concrete repair instructions.
    """
    errors: list[str] = []
    zones = _idf_values(state.idf, "Zone")
    surfaces = _idf_values(state.idf, "BuildingSurface:Detailed")

    if not zones:
        errors.append(
            "Model geometry incomplete: 0 Zone objects. Create thermal zones "
            "before simulation."
        )
    if not surfaces:
        errors.append(
            "Model geometry incomplete: 0 BuildingSurface:Detailed objects. "
            "Create wall/floor/roof surfaces before simulation."
        )

    if zones and surfaces:
        surfaces_by_zone: dict[str, int] = {}
        for surface in surfaces:
            zone_name = getattr(surface, "zone_name", "")
            if zone_name:
                surfaces_by_zone[zone_name] = surfaces_by_zone.get(zone_name, 0) + 1
        for zone in zones:
            name = getattr(zone, "name", "")
            if name and surfaces_by_zone.get(name, 0) == 0:
                errors.append(
                    "Model geometry incomplete: Zone "
                    f"'{name}' has 0 BuildingSurface:Detailed objects. "
                    "Create wall/floor/roof surfaces for this zone before "
                    "simulation."
                )

    return errors


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
        try:
            validation = self.validate_config()
            if not validation.success:
                return ToolResponse(
                    success=False,
                    message="Validation reference errors, cannot run simulation.",
                    data=validation.data,
                )

            geometry_errors = validate_geometry_completeness(self.state)
            if geometry_errors:
                return ToolResponse(
                    success=False,
                    message=("Geometry completeness errors, cannot run simulation."),
                    data={"errors": geometry_errors},
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
