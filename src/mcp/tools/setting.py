from typing import Any, Type
from pydantic import BaseModel
from src.mcp.state import ConfigState
from src.mcp.tools.base import BaseTool
from src.utils.logging import get_logger
from src.validator.data_model import (
    SimulationControlSchema,
    SiteLocationSchema,
    GlobalGeometryRulesSchema,
    RunPeriodSchema,
    OutputVariableDictionarySchema,
    OutputDiagnosticsSchema,
    OutputTableSummaryReportsSchema,
    OutputControlTableStyleSchema,
)

logger = get_logger(__name__)

class SettingTool(BaseTool):
    def __init__(self, state: ConfigState):
        super().__init__(state, "Setting")
        self.mapping: dict[str, tuple[Type[BaseModel], str]] = {
            "SimulationControl": (SimulationControlSchema, "simulation_control"),
            "Site:Location": (SiteLocationSchema, "site_location"),
            "GlobalGeometryRules": (GlobalGeometryRulesSchema, "global_geometry_rules"),
            "RunPeriod": (RunPeriodSchema, "run_period"),
            "Output:VariableDictionary": (OutputVariableDictionarySchema, "output_variable_dictionary"),
            "Output:Diagnostics": (OutputDiagnosticsSchema, "output_diagnostics"),
            "Output:Table:SummaryReports": (OutputTableSummaryReportsSchema, "output_table_summary_reports"),
            "OutputControl:Table:Style": (OutputControlTableStyleSchema, "output_control_table_style"),
        }

    @property
    def storage(self) -> dict[str, BaseModel]:
        storage = {}
        for key, (schema_cls, attr) in self.mapping.items():
            val = getattr(self.state, attr, None)
            if isinstance(val, dict):
                try:
                    model_instance = schema_cls.model_validate(val)
                    setattr(self.state, attr, model_instance)
                    val = model_instance
                except Exception as e:
                    logger.warning(f"Auto-conversion failed for setting '{key}': {e}")
                    pass

            if val is not None:
                storage[key] = val
        return storage

    def _validate_and_create(self, data: dict[str, Any]) -> BaseModel:
        setting_type = data.get("Name")
        if not setting_type or setting_type not in self.mapping:
             raise ValueError(f"Unknown or missing 'Name' for Setting. Supported: {list(self.mapping.keys())}")

        schema_cls, _ = self.mapping[setting_type]
        validation_data = data.copy()
        if "Name" in validation_data:
            validation_data.pop("Name")
        return schema_cls.model_validate(validation_data)

    def _get_name(self, instance: Any) -> str:
        instance_type = type(instance)
        for key, (cls, _) in self.mapping.items():
            if instance_type == cls:
                return key
        return "Unknown"

    def _check_references(self, name: str) -> list[str]:
        return []

    def _add_to_storage(self, instance: Any) -> None:
        key = self._get_name(instance)
        if key != "Unknown":
            attr = self.mapping[key][1]
            setattr(self.state, attr, instance)

    def _remove_from_storage(self, name: str) -> None:
        if name in self.mapping:
            schema_cls, attr = self.mapping[name]
            field = self.state.model_fields.get(attr)
            if field and field.default is None:
                setattr(self.state, attr, None)
            else:
                setattr(self.state, attr, schema_cls())

    def _update_storage(self, name: str, instance: Any) -> None:
        self._add_to_storage(instance)
