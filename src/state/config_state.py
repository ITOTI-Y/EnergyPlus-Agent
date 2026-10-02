from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from idfpy import IDF, IDFBaseModel
from idfpy.models import (
    Building,
    BuildingSurfaceDetailed,
    Construction,
    FenestrationSurfaceDetailed,
    GlobalGeometryRules,
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    RunPeriod,
    ScheduleCompact,
    SimulationControl,
    SiteLocation,
    WindowMaterialSimpleGlazingSystem,
    Zone,
)
from loguru import logger
from pydantic import BaseModel, PrivateAttr

from src.state.defaults import new_model


class ConfigSummary(BaseModel):
    """Summary snapshot of the current EnergyPlus IDF state."""

    building: dict[str, Any] | None = None
    site_location: dict[str, Any] | None = None
    zones_count: int = 0
    materials_count: int = 0
    constructions_count: int = 0
    surfaces_count: int = 0
    fenestrations_count: int = 0
    schedules_count: int = 0
    hvac_thermostats_count: int = 0
    hvac_ideal_loads_count: int = 0
    simulation_control: dict[str, Any] | None = None
    run_period: dict[str, Any] | None = None
    global_geometry_rules: dict[str, Any] | None = None


def _first_dump(idf: IDF, object_type: type[IDFBaseModel]) -> dict[str, Any] | None:
    obj = next(iter(idf.all_of_type(object_type).values()), None)
    return obj.model_dump(exclude_none=True) if obj is not None else None


def _names(idf: IDF, *object_types: type[IDFBaseModel]) -> set[str]:
    return {name for t in object_types for name in idf.all_of_type(t)}


_MATERIAL_TYPES: Final = (
    Material,
    MaterialNoMass,
    MaterialAirGap,
    WindowMaterialSimpleGlazingSystem,
)


class ConfigState(BaseModel):
    """Holds one idfpy model for the agent graph and the MCP server.

    A Pydantic model so that ``AgentState`` can carry it and deep-copy it per
    parallel branch; all content lives in the private IDF.
    """

    _idf: IDF = PrivateAttr(default_factory=new_model)

    @property
    def idf(self) -> IDF:
        return self._idf

    def attach_idf(self, idf: IDF) -> None:
        self._idf = idf

    def clear(self) -> None:
        self._idf = new_model()

    def save_model(self, path: Path) -> Path:
        """Write the model as IDF or epJSON, chosen by the file suffix.

        Raises:
            ValueError: If the suffix is neither ``.idf`` nor ``.epJSON``.
        """
        if path.suffix.lower() not in (".idf", ".epjson"):
            raise ValueError(f"Unsupported model file {path}; use .idf or .epJSON")
        match path.suffix.lower():
            case ".idf":
                self._idf.save(path)
            case _:
                self._idf.save(path, output_type="epjson")
        logger.info("Saved model to {}", path)
        return path

    def load_model(self, path: Path) -> None:
        """Replace the model with an IDF or epJSON file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the suffix is neither ``.idf`` nor ``.epJSON``.
            pydantic.ValidationError: If an epJSON object is invalid.
        """
        match path.suffix.lower():
            case ".idf" | ".epjson":
                self._idf = IDF.load(path, strict=True)
            case _:
                raise ValueError(f"Unsupported model file {path}; use .idf or .epJSON")
        logger.info("Loaded model from {}", path)

    def get_summary(self) -> ConfigSummary:
        idf = self._idf
        return ConfigSummary(
            building=_first_dump(idf, Building),
            site_location=_first_dump(idf, SiteLocation),
            zones_count=len(idf.all_of_type(Zone)),
            materials_count=len(_names(idf, *_MATERIAL_TYPES)),
            constructions_count=len(idf.all_of_type(Construction)),
            surfaces_count=len(idf.all_of_type(BuildingSurfaceDetailed)),
            fenestrations_count=len(idf.all_of_type(FenestrationSurfaceDetailed)),
            schedules_count=len(idf.all_of_type(ScheduleCompact)),
            hvac_thermostats_count=len(idf.all_of_type(HVACTemplateThermostat)),
            hvac_ideal_loads_count=len(
                idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem)
            ),
            simulation_control=_first_dump(idf, SimulationControl),
            run_period=_first_dump(idf, RunPeriod),
            global_geometry_rules=_first_dump(idf, GlobalGeometryRules),
        )
