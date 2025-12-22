from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from src.mcp.interface import ConfigSummary
from src.validator import (
    ConstructionSchema,
    FenestrationSurfaceSchema,
    HVACTemplateThermostatSchema,
    HVACTemplateZoneIdealLoadsAirSystemSchema,
    MaterialSchema,
    RunPeriodSchema,
    ScheduleCompactSchema,
    ScheduleTypeLimitsSchema,
    SimulationControlSchema,
    SiteLocationSchema,
    SurfaceSchema,
    ZoneSchema,
)


class ConfigState(BaseModel):
    zones: dict[str, ZoneSchema] = Field(default_factory=dict, alias="Zone")
    materials: dict[str, MaterialSchema] = Field(default_factory=dict, alias="Material")
    constructions: dict[str, ConstructionSchema] = Field(default_factory=dict, alias="Constructions")
    surfaces: dict[str, SurfaceSchema] = Field(default_factory=dict, alias="BuildingSurface:Detailed")
    fenestrations: dict[str, FenestrationSurfaceSchema] = Field(default_factory=dict, alias="FenestrationSurface:Detailed")

    schedule_type_limits: dict[str, ScheduleTypeLimitsSchema] = Field(
        default_factory=dict, alias="ScheduleTypeLimits"
    )
    schedules: dict[str, ScheduleCompactSchema] = Field(default_factory=dict, alias="Schedule:Compact")

    hvac_thermostats: dict[str, HVACTemplateThermostatSchema] = Field(
        default_factory=dict, alias="HVACTemplate:Thermostat"
    )
    hvac_ideal_loads: dict[str, HVACTemplateZoneIdealLoadsAirSystemSchema] = Field(
        default_factory=dict, alias="HVACTemplate:Zone:IdealLoadsAirSystem"
    )

    simulation_control: SimulationControlSchema | None = Field(default=None, alias="SimulationControl")
    site_location: SiteLocationSchema | None = Field(default=None, alias="Site:Location")
    run_period: RunPeriodSchema | None = Field(default=None, alias="RunPeriod")

    def to_yaml_dict(self) -> list[dict[str, Any]]:
        result = []
        for _, value in self.model_dump().items():
            result.append(value.to_yaml_dict())
        return result

    def export_yaml(self, output_path: str | Path) -> None:
        if isinstance(output_path, str):
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            OmegaConf.save(config=self.to_yaml_dict(), f=f)
        self.logger.info(f"Exported YAML to {output_path}")

    def load_yaml(self, input_path: str | Path) -> None:
        if isinstance(input_path, str):
            input_path = Path(input_path)
        with open(input_path) as f:
            config = OmegaConf.load(f)
            self.model_validate(config)
        self.logger.info(f"Loaded YAML from {input_path}")

    def clear(self) -> None:
        self.zones.clear()
        self.materials.clear()
        self.constructions.clear()
        self.surfaces.clear()
        self.fenestrations.clear()
        self.schedule_type_limits.clear()
        self.schedules.clear()
        self.hvac_thermostats.clear()
        self.hvac_ideal_loads.clear()
        self.simulation_control = None
        self.site_location = None

    def get_summary(self) -> ConfigSummary:
        return ConfigSummary(
            zones_count=len(self.zones),
            materials_count=len(self.materials),
            constructions_count=len(self.constructions),
            surfaces_count=len(self.surfaces),
            fenestrations_count=len(self.fenestrations),
            schedules_count=len(self.schedules),
            hvac_thermostats_count=len(self.hvac_thermostats),
            hvac_ideal_loads_count=len(self.hvac_ideal_loads),
            has_simulation_control=self.simulation_control is not None,
            has_site_location=self.site_location is not None,
            has_run_period=self.run_period is not None,
        )
