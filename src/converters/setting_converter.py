from eppy.modeleditor import IDF
from typing import Dict, Any, Tuple
from src.converters.base_converter import BaseConverter
from src.validator.data_model import (
    VersionSchema,
    SimulationControlSchema,
    TimestepSchema,
    RunPeriodSchema,
    GlobalGeometryRulesSchema,
    SiteLocationSchema,
    OutputVariableDictionarySchema,
    OutputDiagnosticsSchema,
    OutputTableSummaryReportsSchema,
    OutputControlTableStyleSchema,
    OutputVariableSchema,
)


class SettingsConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)

        self.setting_map = {
            "SimulationControl": SimulationControlSchema,
            "Timestep": TimestepSchema,
            "RunPeriod": RunPeriodSchema,
            "GlobalGeometryRules": GlobalGeometryRulesSchema,
            "Site:Location": SiteLocationSchema,
            "Output:VariableDictionary": OutputVariableDictionarySchema,
            "Output:Diagnostics": OutputDiagnosticsSchema,
            "Output:Table:SummaryReports": OutputTableSummaryReportsSchema,
            "OutputControl:Table:Style": OutputControlTableStyleSchema,
            "Output:Variable": OutputVariableSchema,
        }

    def convert(self, data: Dict[str, Any]) -> None:
        self.logger.info("Settings Converter Starting...")

        version_tuple: Tuple[int, ...] = self.idf.idd_version
        global_settings_data = {
            key: data.get(key) for key in self.setting_map if key in data
        }

        try:
            data_to_validate = {
                "version_data": {"version": version_tuple},
                "global_settings_data": global_settings_data,
            }
            validated_data = self.validate(data_to_validate)
            self._add_to_idf(validated_data)
            self.state["success"] += 1
        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(
                f"Error during settings conversion process: {e}", exc_info=True
            )

    def validate(self, data: Dict) -> Dict:
        self.logger.info("Validating global settings...")

        val_version_data = VersionSchema.model_validate(data.get("version_data", {}))

        validated_settings = {}
        raw_global_settings = data.get("global_settings_data", {})

        for idf_key, setting_data in raw_global_settings.items():
            if setting_data is None:
                continue

            schema = self.setting_map.get(idf_key)
            if not schema:
                self.logger.warning(
                    f"No schema found for '{idf_key}', skipping validation for this item."
                )
                continue

            try:
                if isinstance(setting_data, list):
                    validated_settings[idf_key] = [
                        schema.model_validate(item) for item in setting_data
                    ]
                else:
                    validated_settings[idf_key] = schema.model_validate(setting_data)
            except Exception as e:
                self.logger.error(
                    f"Validation failed for '{idf_key}': {e}", exc_info=True
                )
                raise

        return {
            "version_info": val_version_data,
            "validated_settings": validated_settings,
        }

    def _add_to_idf(self, val_data: Dict) -> None:
        version_info = val_data.get("version_info")
        settings_to_add = val_data.get("validated_settings", {})

        if version_info and not self.idf.idfobjects.get("Version"):
            self.logger.info(f"Adding Version object '{version_info.version}' to IDF.")
            self.idf.newidfobject("Version", Version_Identifier=version_info.version)

        for idf_key, validated_model_or_list in settings_to_add.items():
            items_to_process = (
                validated_model_or_list
                if isinstance(validated_model_or_list, list)
                else [validated_model_or_list]
            )
            for validated_model in items_to_process:
                self._add_single_object_to_idf(idf_key, validated_model)

    def _add_single_object_to_idf(self, idf_key: str, validated_model) -> None:
        data_dict = validated_model.model_dump(exclude_none=True)

        if (
            idf_key != "Output:Variable"
            and len(self.idf.idfobjects.get(idf_key, [])) > 0
        ):
            self.logger.warning(
                f"Object of type '{idf_key}' already exists. Skipping addition."
            )
            return

        self.idf.newidfobject(idf_key, **data_dict)
        self.logger.success(f"Added setting '{idf_key}' to IDF.")
