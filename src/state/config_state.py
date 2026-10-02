from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from idfpy import IDF, IDFBaseModel, RefError
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.hvac_templates import (
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
)
from idfpy.models.internal_gains import Lights, People
from idfpy.models.location import RunPeriod, SiteLocation
from idfpy.models.schedules import ScheduleCompact, ScheduleTypeLimits
from idfpy.models.simulation import Building, SimulationControl
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
    GlobalGeometryRules,
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


def missing_references(idf: IDF, instance: Any, object_name: str) -> list[str]:
    """Validate one instance's cross-references before it is added to the IDF.

    Wraps idfpy's private per-object validation (``IDF.validate()`` only
    covers objects already added); the instance itself need not be in the
    IDF, only the referenced providers must already exist. Sole call site
    of the private API — update here if idfpy changes it.
    """
    errors: list[RefError] = []
    idf._validate_obj_refs(object_name, instance, errors)
    return [f"{err.field_name}: {err.detail}" for err in errors]


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
        match path.suffix.lower():
            case ".idf":
                self._idf.save(path)
            case ".epjson":
                self._idf.save(path, output_type="epjson")
            case _:
                raise ValueError(f"Unsupported model file {path}; use .idf or .epJSON")
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
            case ".idf":
                self._idf = IDF.load(path)
            case ".epjson":
                # IDF.load skips invalid epJSON objects; strict parsing keeps
                # a load from silently dropping part of the model.
                self._idf = IDF.from_dict(json.loads(path.read_text(encoding="utf-8")))
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

    def validate_references(self) -> list[str]:
        idf = self._idf
        errors: list[str] = []
        material_names = _names(idf, *_MATERIAL_TYPES)
        construction_names = _names(idf, Construction)
        surface_names = _names(idf, BuildingSurfaceDetailed)
        zone_names = _names(idf, Zone)
        schedule_names = _names(idf, ScheduleCompact)
        thermostat_names = _names(idf, HVACTemplateThermostat)
        type_limits_names = _names(idf, ScheduleTypeLimits)

        for schedule in idf.all_of_type(ScheduleCompact).values():
            limits = schedule.schedule_type_limits_name
            if limits and limits not in type_limits_names:
                errors.append(
                    f"Schedule '{schedule.name}' references type limits '{limits}' which does not exist."
                )

        for const in idf.all_of_type(Construction).values():
            for layer in (
                const.outside_layer,
                const.layer_2,
                const.layer_3,
                const.layer_4,
                const.layer_5,
                const.layer_6,
                const.layer_7,
                const.layer_8,
                const.layer_9,
                const.layer_10,
            ):
                if layer and layer not in material_names:
                    errors.append(
                        f"Construction '{const.name}' references material '{layer}' which does not exist."
                    )

        for surface in idf.all_of_type(BuildingSurfaceDetailed).values():
            if surface.construction_name not in construction_names:
                errors.append(
                    f"Surface '{surface.name}' references construction '{surface.construction_name}' which does not exist."
                )
            if surface.zone_name not in zone_names:
                errors.append(
                    f"Surface '{surface.name}' references zone '{surface.zone_name}' which does not exist."
                )

        for fen in idf.all_of_type(FenestrationSurfaceDetailed).values():
            if fen.construction_name not in construction_names:
                errors.append(
                    f"Fenestration '{fen.name}' references construction '{fen.construction_name}' which does not exist."
                )
            if fen.building_surface_name not in surface_names:
                errors.append(
                    f"Fenestration '{fen.name}' references building surface '{fen.building_surface_name}' which does not exist."
                )

        for ils in idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).values():
            if ils.zone_name not in zone_names:
                errors.append(
                    f"Ideal load system references zone '{ils.zone_name}' which does not exist."
                )
            if ils.template_thermostat_name not in thermostat_names:
                errors.append(
                    f"Ideal load system references thermostat '{ils.template_thermostat_name}' which does not exist."
                )
            avail = ils.system_availability_schedule_name
            if avail and avail not in schedule_names:
                errors.append(
                    f"Ideal load system for zone '{ils.zone_name}' references availability schedule '{avail}' which does not exist."
                )

        for thermostat in idf.all_of_type(HVACTemplateThermostat).values():
            if thermostat.heating_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references heating setpoint schedule '{thermostat.heating_setpoint_schedule_name}' which does not exist."
                )
            if thermostat.cooling_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references cooling setpoint schedule '{thermostat.cooling_setpoint_schedule_name}' which does not exist."
                )

        for people in idf.all_of_type(People).values():
            zone = people.zone_or_zonelist_or_space_or_spacelist_name
            if zone and zone not in zone_names:
                errors.append(
                    f"People '{people.name}' references zone '{zone}' which does not exist."
                )
            for sched in (
                people.number_of_people_schedule_name,
                people.activity_level_schedule_name,
                people.work_efficiency_schedule_name,
                people.clothing_insulation_calculation_method_schedule_name,
                people.clothing_insulation_schedule_name,
                people.air_velocity_schedule_name,
                people.ankle_level_air_velocity_schedule_name,
            ):
                if sched and sched not in schedule_names:
                    errors.append(
                        f"People '{people.name}' references schedule '{sched}' which does not exist."
                    )

        for light in idf.all_of_type(Lights).values():
            zone = light.zone_or_zonelist_or_space_or_spacelist_name
            if zone and zone not in zone_names:
                errors.append(
                    f"Lights '{light.name}' references zone '{zone}' which does not exist."
                )
            if light.schedule_name and light.schedule_name not in schedule_names:
                errors.append(
                    f"Lights '{light.name}' references schedule '{light.schedule_name}' which does not exist."
                )

        return errors
