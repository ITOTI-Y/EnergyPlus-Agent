from typing import Any

from idfpy import IDF
from idfpy.models.location import RunPeriod, SiteLocation
from idfpy.models.outputs import (
    OutputControlTableStyle,
    OutputDiagnostics,
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReports,
    OutputTableSummaryReportsReportsItem,
    OutputVariable,
    OutputVariableDictionary,
)
from idfpy.models.simulation import Building, SimulationControl, Timestep, Version
from idfpy.models.thermal_zones import GlobalGeometryRules

from src.converters.base_converter import BaseConverter
from src.validator.data_model import (
    GlobalGeometryRulesSchema,
    OutputControlTableStyleSchema,
    OutputDiagnosticsSchema,
    OutputTableSummaryReportsSchema,
    OutputVariableDictionarySchema,
    OutputVariableSchema,
    RunPeriodSchema,
    SimulationControlSchema,
    SiteLocationSchema,
    TimestepSchema,
    VersionSchema,
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

        self.apply_function_map = {
            "SimulationControl": self._simulation_control_apply,
            "Timestep": self._timestep_apply,
            "RunPeriod": self._run_period_apply,
            "GlobalGeometryRules": self._global_geometry_rules_apply,
            "Site:Location": self._site_location_apply,
            "Output:VariableDictionary": self._output_variable_dictionary_apply,
            "Output:Diagnostics": self._output_diagnostics_apply,
            "Output:Table:SummaryReports": self._output_table_summary_reports_apply,
            "OutputControl:Table:Style": self._output_control_table_style_apply,
            "Output:Variable": self._output_variable_apply,
        }

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("Settings Converter Starting...")

        version_str: str = self.idf.version
        global_settings_data = {
            key: data.get(key) for key in self.setting_map if key in data
        }

        try:
            data_to_validate = {
                "version_data": {"version": version_str},
                "global_settings_data": global_settings_data,
            }
            validated_data = self.validate(data_to_validate)
            self._add_to_idf(validated_data)
            self.state["success"] += 1
        except Exception:
            self.state["failed"] += 1
            self.logger.exception("Error during settings conversion process")

    def validate(self, data: dict) -> dict:
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
                    "No schema found for '{}', skipping validation for this item.",
                    idf_key,
                )
                continue

            try:
                if isinstance(setting_data, list):
                    validated_settings[idf_key] = [
                        schema.model_validate(item) for item in setting_data
                    ]
                else:
                    validated_settings[idf_key] = schema.model_validate(setting_data)
            except Exception:
                self.logger.exception("Validation failed for '{}'", idf_key)
                raise

        return {
            "version_info": val_version_data,
            "validated_settings": validated_settings,
        }

    def _add_to_idf(self, val_data: dict) -> None:
        version_info = val_data.get("version_info")
        settings_to_add = val_data.get("validated_settings", {})

        if version_info and not self.idf.all_of_type("Version"):
            self.logger.info("Adding Version object '{}' to IDF.", version_info.version)
            self.idf.add(Version(version_identifier=version_info.version))

        for idf_key, validated_model_or_list in settings_to_add.items():
            items_to_process = (
                validated_model_or_list
                if isinstance(validated_model_or_list, list)
                else [validated_model_or_list]
            )
            for validated_model in items_to_process:
                self._add_single_object_to_idf(idf_key, validated_model)

    def _add_single_object_to_idf(self, idf_key: str, validated_model) -> None:
        if (
            idf_key != "Output:Variable"
            and len(self.idf.all_of_type(idf_key)) > 0
        ):
            self.logger.warning(
                "Object of type '{}' already exists. Skipping addition.", idf_key
            )
            return

        apply_function = self.apply_function_map.get(idf_key)
        if apply_function:
            apply_function(validated_model)
        else:
            self.logger.error("No apply function found for '{}'", idf_key)

    def _simulation_control_apply(self, model: SimulationControlSchema) -> None:
        self.idf.add(SimulationControl(
            do_zone_sizing_calculation=model.do_zone_sizing_calculation,
            do_system_sizing_calculation=model.do_system_sizing_calculation,
            do_plant_sizing_calculation=model.do_plant_sizing_calculation,
            run_simulation_for_sizing_periods=model.run_simulation_for_sizing_periods,
            run_simulation_for_weather_file_run_periods=model.run_simulation_for_weather_file_run_periods,
            do_hvac_sizing_simulation_for_sizing_periods=model.do_hvac_sizing_simulation_for_sizing_periods,
            maximum_number_of_hvac_sizing_simulation_passes=model.maximum_number_of_hvac_sizing_simulation_passes,
        ))
        self.logger.success("Added setting 'SimulationControl' to IDF.")

    def _timestep_apply(self, model: TimestepSchema) -> None:
        self.idf.add(Timestep(
            number_of_timesteps_per_hour=model.number_of_timesteps_per_hour,
        ))
        self.logger.success("Added setting 'Timestep' to IDF.")

    def _run_period_apply(self, model: RunPeriodSchema) -> None:
        kwargs: dict[str, Any] = {
            "name": model.name,
            "begin_month": model.begin_month,
            "begin_day_of_month": model.begin_day_of_month,
            "end_month": model.end_month,
            "end_day_of_month": model.end_day_of_month,
        }
        optional: dict[str, Any] = {
            "begin_year": model.begin_year,
            "end_year": model.end_year,
            "day_of_week_for_start_day": model.day_of_week_for_start_day,
            "use_weather_file_holidays_and_special_days": model.use_weather_file_holidays_and_special_days,
            "use_weather_file_daylight_saving_period": model.use_weather_file_daylight_saving_period,
            "apply_weekend_holiday_rule": model.apply_weekend_holiday_rule,
            "use_weather_file_rain_indicators": model.use_weather_file_rain_indicators,
            "use_weather_file_snow_indicators": model.use_weather_file_snow_indicators,
        }
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        self.idf.add(RunPeriod(**kwargs))
        self.logger.success("Added setting 'RunPeriod' to IDF.")

    def _global_geometry_rules_apply(self, model: GlobalGeometryRulesSchema) -> None:
        self.idf.add(GlobalGeometryRules(
            starting_vertex_position=model.starting_vertex_position,
            vertex_entry_direction=model.vertex_entry_direction,
            coordinate_system=model.coordinate_system,
        ))
        self.logger.success("Added setting 'GlobalGeometryRules' to IDF.")

    def _site_location_apply(self, model: SiteLocationSchema) -> None:
        self.idf.add(SiteLocation(
            name=model.name,
            latitude=model.latitude,
            longitude=model.longitude,
            time_zone=model.time_zone,
            elevation=model.elevation,
        ))
        self.logger.success("Added setting 'Site:Location' to IDF.")

    def _output_variable_dictionary_apply(
        self, model: OutputVariableDictionarySchema
    ) -> None:
        self.idf.add(OutputVariableDictionary(key_field=model.key_field))
        self.logger.success("Added setting 'Output:VariableDictionary' to IDF.")

    def _output_diagnostics_apply(self, model: OutputDiagnosticsSchema) -> None:
        self.idf.add(OutputDiagnostics(
            diagnostics=[OutputDiagnosticsDiagnosticsItem(key=model.key_1)],
        ))
        self.logger.success("Added setting 'Output:Diagnostics' to IDF.")

    def _output_table_summary_reports_apply(
        self, model: OutputTableSummaryReportsSchema
    ) -> None:
        self.idf.add(OutputTableSummaryReports(
            reports=[OutputTableSummaryReportsReportsItem(report_name=model.report_1_name)],
        ))
        self.logger.success("Added setting 'Output:Table:SummaryReports' to IDF.")

    def _output_control_table_style_apply(
        self, model: OutputControlTableStyleSchema
    ) -> None:
        self.idf.add(OutputControlTableStyle(
            column_separator=model.column_separator,
            unit_conversion=model.unit_conversion,
        ))
        self.logger.success("Added setting 'OutputControl:Table:Style' to IDF.")

    def _output_variable_apply(self, model: OutputVariableSchema) -> None:
        self.idf.add(OutputVariable(
            key_value=model.key_value,
            variable_name=model.variable_name,
            reporting_frequency=model.reporting_frequency,
        ))
        self.logger.success("Added setting 'Output:Variable' to IDF.")
