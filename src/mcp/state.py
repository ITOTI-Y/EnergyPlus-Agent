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
    HVACSchema,
    MaterialSchema,
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

    schedules: ScheduleCollectionSchema | None = Field(
        default=None, alias="ScheduleCollection"
    )

    hvac: HVACSchema | None = Field(default=None, alias="HVAC")

    simulation_control: SimulationControlSchema = Field(
        default_factory=SimulationControlSchema, alias="SimulationControl"
    )
    run_period: RunPeriodSchema | None = Field(default=None, alias="RunPeriod")

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
        self.schedules = None
        self.hvac = None
        self.simulation_control = SimulationControlSchema()

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
            has_simulation_control=self.simulation_control is not None,
            has_site_location=self.site_location is not None,
            has_run_period=self.run_period is not None,
        )
