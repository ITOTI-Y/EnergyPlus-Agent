from pathlib import Path
from typing import Any

from loguru import logger
from omegaconf import OmegaConf
from pydantic import Field

from src.mcp.interface import ConfigSummary
from src.validator import (
    BaseSchema,
    BuildingSchema,
    ConstructionSchema,
    FenestrationSurfaceSchema,
    GlobalGeometryRulesSchema,
    HVACSchema,
    MaterialSchema,
    OutputControlTableStyleSchema,
    OutputDiagnosticsSchema,
    OutputTableSummaryReportsSchema,
    OutputVariableDictionarySchema,
    OutputVariableSchema,
    RunPeriodSchema,
    ScheduleCollectionSchema,
    SimulationControlSchema,
    SiteLocationSchema,
    SurfaceSchema,
    ZoneSchema,
)


class ConfigState(BaseSchema):
    building: BuildingSchema | None = Field(default=None, alias="Building")
    site_location: SiteLocationSchema | None = Field(
        default=None, alias="Site:Location"
    )

    zones: list[ZoneSchema] = Field(default_factory=list, alias="Zone")
    materials: list[MaterialSchema] = Field(default_factory=list, alias="Material")
    constructions: list[ConstructionSchema] = Field(
        default_factory=list, alias="Construction"
    )
    surfaces: list[SurfaceSchema] = Field(
        default_factory=list, alias="BuildingSurface:Detailed"
    )
    fenestrations: list[FenestrationSurfaceSchema] = Field(
        default_factory=list, alias="FenestrationSurface:Detailed"
    )

    schedules: ScheduleCollectionSchema = Field(
        default_factory=ScheduleCollectionSchema, alias="Schedule"
    )

    hvac: HVACSchema = Field(default_factory=HVACSchema, alias="HVAC")

    simulation_control: SimulationControlSchema = Field(
        default_factory=SimulationControlSchema, alias="SimulationControl"
    )
    global_geometry_rules: GlobalGeometryRulesSchema = Field(
        default_factory=GlobalGeometryRulesSchema, alias="GlobalGeometryRules"
    )
    run_period: RunPeriodSchema = Field(default_factory=RunPeriodSchema, alias="RunPeriod")

    output_variable_dictionary: OutputVariableDictionarySchema = Field(
        default_factory=dict, alias="Output:VariableDictionary"
    )
    output_diagnostics: OutputDiagnosticsSchema = Field(
        default_factory=dict, alias="Output:Diagnostics"
    )
    output_table_summary_reports: OutputTableSummaryReportsSchema = Field(
        default_factory=dict, alias="Output:Table:SummaryReports"
    )
    output_variable: list[OutputVariableSchema] = Field(
        default_factory=list, alias="Output:Variable"
    )
    output_control_table_style: OutputControlTableStyleSchema = Field(
        default_factory=dict, alias="OutputControl:Table:Style"
    )

    def to_yaml_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True, serialize_as_any=True)

    def export_yaml(self, output_path: str | Path) -> None:
        if isinstance(output_path, str):
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            OmegaConf.save(config=self.to_yaml_dict(), f=f)
        logger.info(f"Exported YAML to {output_path}")

    @classmethod
    def load_yaml(cls, input_path: str | Path) -> "ConfigState":
        if isinstance(input_path, str):
            input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"YAML file not found: {input_path}")
        with open(input_path) as f:
            config = OmegaConf.load(f)
            logger.info(f"Loaded YAML from {input_path}")
            return cls.model_validate(config)

    def clear(self) -> None:
        self.building = None
        self.site_location = None
        self.zones.clear()
        self.materials.clear()
        self.constructions.clear()
        self.surfaces.clear()
        self.fenestrations.clear()
        self.schedules.schedule_type_limits.clear()
        self.schedules.schedules.clear()
        self.hvac.thermostats.clear()
        self.hvac.ideal_loads_systems.clear()
        self.global_geometry_rules = GlobalGeometryRulesSchema()
        self.simulation_control = SimulationControlSchema()
        self.run_period = RunPeriodSchema()
        self.output_variable_dictionary = OutputVariableDictionarySchema()
        self.output_diagnostics = OutputDiagnosticsSchema()
        self.output_table_summary_reports = OutputTableSummaryReportsSchema()
        self.output_variable = []
        self.output_control_table_style = OutputControlTableStyleSchema()

    def update_from(self, other: "ConfigState") -> None:
        """Update all fields from another ConfigState instance."""
        for field_name in self.model_fields:
            setattr(self, field_name, getattr(other, field_name))

    def get_summary(self) -> ConfigSummary:
        return ConfigSummary(
            building=self.building if self.building else None,
            site_location=self.site_location if self.site_location else None,
            zones_count=len(self.zones),
            materials_count=len(self.materials),
            constructions_count=len(self.constructions),
            surfaces_count=len(self.surfaces),
            fenestrations_count=len(self.fenestrations),
            schedules_count=len(self.schedules.schedules) if self.schedules else 0,
            hvac_thermostats_count=len(self.hvac.thermostats) if self.hvac else 0,
            hvac_ideal_loads_count=len(self.hvac.ideal_loads_systems)
            if self.hvac
            else 0,
            simulation_control=self.simulation_control
            if self.simulation_control
            else None,
            run_period=self.run_period if self.run_period else None,
            global_geometry_rules=self.global_geometry_rules
            if self.global_geometry_rules
            else None,
        )

    def validate_references(self) -> list[str]:
        errors = []

        material_names = [material.name for material in self.materials]
        construction_names = [construction.name for construction in self.constructions]
        surface_names = [surface.name for surface in self.surfaces]
        zone_names = [zone.name for zone in self.zones]
        schedule_names = (
            [schedule.name for schedule in self.schedules.schedules]
            if self.schedules and self.schedules.schedules
            else []
        )
        thermostat_names = (
            [thermostat.name for thermostat in self.hvac.thermostats]
            if self.hvac and self.hvac.thermostats
            else []
        )

        for const in self.constructions if self.constructions else []:
            for layer in const.layers:
                if layer not in material_names:
                    errors.append(
                        f"Construction '{const.name}' references material '{layer}' which does not exist."
                    )

        for surface in self.surfaces if self.surfaces else []:
            if surface.construction_name not in construction_names:
                errors.append(
                    f"Surface '{surface.name}' references construction '{surface.construction_name}' which does not exist."
                )
            if surface.zone_name not in zone_names:
                errors.append(
                    f"Surface '{surface.name}' references zone '{surface.zone_name}' which does not exist."
                )

        for fenestration in self.fenestrations if self.fenestrations else []:
            if fenestration.construction_name not in construction_names:
                errors.append(
                    f"Fenestration '{fenestration.name}' references construction '{fenestration.construction_name}' which does not exist."
                )
            if fenestration.building_surface_name not in surface_names:
                errors.append(
                    f"Fenestration '{fenestration.name}' references building surface '{fenestration.building_surface_name}' which does not exist."
                )

        for ils in (
            self.hvac.ideal_loads_systems
            if self.hvac and self.hvac.ideal_loads_systems
            else []
        ):
            if ils.zone_name not in zone_names:
                errors.append(
                    f"Ideal load system references zone '{ils.zone_name}' which does not exist."
                )
            if ils.template_thermostat_name not in thermostat_names:
                errors.append(
                    f"Ideal load system references thermostat '{ils.template_thermostat_name}' which does not exist in HVAC schema."
                )

        for thermostat in (
            self.hvac.thermostats if self.hvac and self.hvac.thermostats else []
        ):
            if thermostat.heating_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references heating setpoint schedule '{thermostat.heating_setpoint_schedule_name}' which does not exist."
                )
            if thermostat.cooling_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references cooling setpoint schedule '{thermostat.cooling_setpoint_schedule_name}' which does not exist."
                )

        return errors
