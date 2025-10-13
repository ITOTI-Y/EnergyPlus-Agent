from typing import Tuple, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.utils.logging import get_logger

logger = get_logger(__name__)


class IDDField:
    def __init__(self, data: List[Dict] | Dict):
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, list):
                    if len(obj) > 0 and isinstance(obj[0], dict):
                        obj_name = obj[0].get("idfobj", None)
                    else:
                        continue
                    if obj_name:
                        obj_name = self._clean_key(obj_name)
                        setattr(self, obj_name, IDDField(obj[1:]))
                elif isinstance(obj, dict):
                    field_name = obj.get("field", None)
                    if field_name:
                        if (
                            isinstance(field_name, (list, tuple))
                            and len(field_name) > 0
                        ):
                            field_name = self._clean_key(field_name[0])
                        elif isinstance(field_name, str):
                            field_name = self._clean_key(field_name)
                        else:
                            continue
                        setattr(self, field_name, IDDField(obj))
        elif isinstance(data, dict):
            for key, value in data.items():
                key = self._clean_key(key)
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                setattr(self, key, value)

    def _clean_key(self, key: str) -> str:
        for i in [" ", "-", "/", ":"]:
            key = key.replace(i, "_")
        return key


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,  # 支持从对象创建模型
        validate_assignment=True,  # 赋值时验证
        arbitrary_types_allowed=True,  # 允许任意类型
        str_strip_whitespace=True,  # 自动去除字符串空格
        use_enum_values=True,  # 使用枚举值
        populate_by_name=True,  # 允许通过字段名填充
        extra="ignore",  # 忽略额外字段
    )

    _idf_field: IDDField = IDDField({})

    @classmethod
    def set_idf_field(cls, idf_field: IDDField):
        cls._idf_field = idf_field

    @property
    def idf_field(self) -> IDDField:
        return self._idf_field

    @staticmethod
    def validate_choice_field(value: str, valid_choices: list, field_name: str) -> str:
        choice_mapping = {choice.lower(): choice for choice in valid_choices}
        value_lower = value.lower()

        if value_lower not in choice_mapping:
            logger.error(
                f"{field_name} '{value}' is not a valid choice. Valid choices are: {valid_choices}."
            )
            raise ValueError(f"{field_name} must be one of {valid_choices}.")

        if value not in valid_choices:
            logger.warning(
                f"{field_name} '{value}' is not in the standard casing. Using '{choice_mapping[value_lower]}' instead."
            )
        return choice_mapping[value_lower]


class BuildingSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Building name")
    north_axis: float = Field(
        0.0, alias="North Axis", description="Building north axis in degrees"
    )
    terrain: str = Field("Suburbs", alias="Terrain", description="Terrain type")
    loads_convergence_tolerance_value: float = Field(
        0.04,
        alias="Loads Convergence Tolerance Value",
        description="Loads convergence tolerance value",
    )
    temperature_convergence_tolerance_value: float = Field(
        0.4,
        alias="Temperature Convergence Tolerance Value",
        description="Temperature convergence tolerance value",
    )
    solar_distribution: str = Field(
        "FullExterior", alias="Solar Distribution", description="Solar distribution"
    )
    maximum_number_of_warmup_days: int = Field(
        25,
        alias="Maximum Number of Warmup Days",
        description="Maximum number of warmup days",
    )
    minimum_number_of_warmup_days: int = Field(
        0,
        alias="Minimum Number of Warmup Days",
        description="Minimum number of warmup days",
    )

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v

    @field_validator("north_axis")
    def validate_north_axis(cls, v):
        if not (0 <= v < 360):
            raise ValueError("North Axis must be in [0, 360).")
        return v

    @field_validator("terrain")
    def validate_terrain(cls, v):
        valid_terrains = {"Suburbs", "Country", "City", "Ocean", "Urban"}
        if v not in valid_terrains:
            raise ValueError(f"Terrain must be one of {valid_terrains}.")
        return v

    @field_validator(
        "loads_convergence_tolerance_value", "temperature_convergence_tolerance_value"
    )
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive.")
        return v

    @field_validator("solar_distribution")
    def validate_solar_distribution(cls, v):
        valid_distribution = {
            "FullExterior",
            "MinimalShadowing",
            "FullInteriorAndExterior",
            "FullExteriorWithReflections",
            "FullInteriorAndExteriorWithReflections",
        }
        if v not in valid_distribution:
            raise ValueError(f"Solar Distribution must be one of {valid_distribution}.")
        return v

    @field_validator("maximum_number_of_warmup_days", "minimum_number_of_warmup_days")
    def validate_warmup_days(cls, v):
        if v < 0:
            raise ValueError("Warmup days must be non-negative.")
        return v


class VersionSchema(BaseSchema):
    version: str | Tuple | List = Field(
        ..., alias="Version Identifier", description="Version identifier"
    )

    @field_validator("version")
    def validate_version(cls, v):
        if not v:
            raise ValueError("Version Identifier must not be empty.")
        if isinstance(v, (list, tuple)):
            return ".".join([str(i) for i in v])
        if isinstance(v, str):
            return v
        raise ValueError(
            "Version Identifier must be a string or a tuple/list of integers."
        )


class ZoneSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Zone name")
    direction_of_relative_north: Optional[float] = Field(
        0.0,
        alias="Direction of Relative North",
        description="Direction of relative north in degrees",
    )
    x_origin: float = Field(0.0, alias="X Origin", description="X origin coordinate")
    y_origin: float = Field(0.0, alias="Y Origin", description="Y origin coordinate")
    z_origin: float = Field(0.0, alias="Z Origin", description="Z origin coordinate")
    type: int = Field(
        1, alias="Type", description="Zone type is currently unused in EnergyPlus"
    )
    multiplier: int = Field(1, alias="Multiplier", description="Zone multiplier", ge=1)
    ceiling_height: str | float = Field(
        "autocalculate",
        alias="Ceiling Height",
        description="Ceiling height in meters or 'autocalculate'",
    )
    volume: str | float = Field(
        "autocalculate",
        alias="Volume",
        description="Zone volume in cubic meters or 'autocalculate'",
    )
    floor_area: str | float = Field(
        "autocalculate",
        alias="Floor Area",
        description="Zone floor area in square meters or 'autocalculate'",
    )
    zone_inside_convection_algorithm: str = Field(
        "TARP",
        alias="Zone Inside Convection Algorithm",
        description="Zone inside convection algorithm",
    )
    zone_outside_convection_algorithm: str = Field(
        "DOE-2",
        alias="Zone Outside Convection Algorithm",
        description="Zone outside convection algorithm",
    )
    part_of_total_floor_area: str = Field(
        "Yes", alias="Part of Total Floor Area", description="Part of total floor area"
    )

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v

    @field_validator("direction_of_relative_north")
    def validate_direction_of_relative_north(cls, v):
        if v is not None and not (0 <= v < 360):
            raise ValueError("Direction of Relative North must be in [0, 360).")
        elif v is None:
            return 0.0
        return v

    @field_validator("x_origin", "y_origin", "z_origin")
    def validate_origin(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Origin coordinates must be numeric.")
        return v

    @field_validator("type")
    def validate_type(cls, v):
        if v < 0:
            raise ValueError("Zone Type must be non-negative.")
        return v

    @field_validator("multiplier")
    def validate_multiplier(cls, v):
        if v < 1:
            raise ValueError("Multiplier must be at least 1.")
        return v

    @field_validator("ceiling_height", "volume", "floor_area")
    def validate_autocalculate_or_positive(cls, v):
        if isinstance(v, str) and v.lower() == "autocalculate":
            return "autocalculate"
        try:
            fv = float(v)
            if fv <= 0:
                raise ValueError("Value must be positive or 'autocalculate'.")
            return fv
        except (TypeError, ValueError):
            raise ValueError("Value must be a number or 'autocalculate'.")

    @field_validator("zone_inside_convection_algorithm")
    def validate_zone_inside_convection_algorithm(cls, v):
        valid_algorithms = {
            "Simple",
            "TARP",
            "CeilingDiffuser",
            "AdaptiveConvectionAlgorithm",
            "TrombeWall",
            "ASTMC1340",
        }
        if v not in valid_algorithms:
            raise ValueError(
                f"Zone Inside Convection Algorithm must be one of {valid_algorithms}."
            )
        return v

    @field_validator("zone_outside_convection_algorithm")
    def validate_zone_outside_convection_algorithm(cls, v):
        valid_algorithms = {
            "Simple",
            "TARP",
            "DOE-2",
            "MoWiTT",
            "AdaptiveConvectionAlgorithm",
        }
        if v not in valid_algorithms:
            raise ValueError(
                f"Zone Outside Convection Algorithm must be one of {valid_algorithms}."
            )
        return v

    @field_validator("part_of_total_floor_area")
    def validate_part_of_total_floor_area(cls, v):
        valid_options = {"Yes", "No"}
        if v not in valid_options:
            raise ValueError(
                f"Part of Total Floor Area must be one of {valid_options}."
            )
        return v


class SurfaceSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Surface name")
    surface_type: str = Field(..., alias="Surface Type", description="Type of surface")
    construction_name: str = Field(
        ..., alias="Construction Name", description="Name of the construction"
    )
    zone_name: str = Field(
        ..., alias="Zone Name", description="Name of the associated zone"
    )
    space_name: Optional[str] = Field(
        None, alias="Space Name", description="Name of the associated space"
    )
    outside_boundary_condition: str = Field(
        ...,
        alias="Outside Boundary Condition",
        description="Outside boundary condition",
    )
    outside_boundary_condition_object: Optional[str] = Field(
        None,
        alias="Outside Boundary Condition Object",
        description="Outside boundary condition object",
    )
    sun_exposure: str = Field("NoSun", alias="Sun Exposure", description="Sun exposure")
    wind_exposure: str = Field(
        "NoWind", alias="Wind Exposure", description="Wind exposure"
    )
    view_factor_to_ground: str | float = Field(
        "autocalculate",
        alias="View Factor to Ground",
        description="View factor to ground or 'autocalculate'",
    )
    vertices: List[dict[str, float]] = Field(
        ..., alias="Vertices", description="List of vertices defining the surface"
    )

    @field_validator("name", "construction_name", "zone_name")
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("This field must not be empty.")
        return v

    @field_validator("surface_type")
    def validate_surface_type(cls, v):
        valid_types = getattr(
            cls._idf_field, "BuildingSurface_Detailed"
        ).Surface_Type.key
        if v not in valid_types:
            raise ValueError(f"Surface Type must be one of {valid_types}.")
        return v

    @field_validator("outside_boundary_condition")
    def validate_outside_boundary_condition(cls, v):
        valid_conditions = getattr(
            cls._idf_field, "BuildingSurface_Detailed"
        ).Outside_Boundary_Condition.key
        if v not in valid_conditions:
            raise ValueError(
                f"Outside Boundary Condition must be one of {valid_conditions}."
            )
        return v

    @field_validator("sun_exposure")
    def validate_sun_exposure(cls, v):
        valid_exposures = getattr(
            cls._idf_field, "BuildingSurface_Detailed"
        ).Sun_Exposure.key
        if v not in valid_exposures:
            raise ValueError(f"Sun Exposure must be one of {valid_exposures}.")
        return v

    @field_validator("wind_exposure")
    def validate_wind_exposure(cls, v):
        valid_exposures = getattr(
            cls._idf_field, "BuildingSurface_Detailed"
        ).Wind_Exposure.key
        if v not in valid_exposures:
            raise ValueError(f"Wind Exposure must be one of {valid_exposures}.")
        return v

    @field_validator("view_factor_to_ground")
    def validate_view_factor_to_ground(cls, v):
        if isinstance(v, str) and v.lower() == "autocalculate":
            return "autocalculate"
        try:
            fv = float(v)
            if not (0.0 <= fv <= 1.0):
                raise ValueError("View Factor to Ground must be between 0.0 and 1.0.")
            return fv
        except (TypeError, ValueError):
            raise ValueError(
                "View Factor to Ground must be a number between 0.0 and 1.0 or 'autocalculate'."
            )

    @field_validator("vertices")
    def validate_vertices(cls, v):
        if not isinstance(v, list) or len(v) < 3:
            raise ValueError(
                "Vertices must be a list with at least three vertex dictionaries."
            )
        for vertex in v:
            if not (
                isinstance(vertex, dict)
                and all(k in vertex for k in ("X", "Y", "Z"))
                and all(isinstance(vertex[k], (int, float)) for k in ("X", "Y", "Z"))
            ):
                raise ValueError(
                    "Each vertex must be a dictionary with numeric keys 'X', 'Y', and 'Z'."
                )
        # 还需要添加一个验证，确保顶点是按顺序排列的，且形成一个闭合多边形

        return v

    @model_validator(mode="after")
    def validate_boundary_condition_object(self):
        needs_obj = {"Surface", "OtherSideCoefficients", "OtherSideConditionsModel"}
        if self.outside_boundary_condition in needs_obj:
            if not self.outside_boundary_condition_object:
                raise ValueError(
                    f"Outside Boundary Condition Object is required when "
                    f"Outside Boundary Condition is '{self.outside_boundary_condition}'."
                )
        return self


class SimulationControlSchema(BaseSchema):
    Do_Zone_Sizing_Calculation: str | bool = Field(
        "No", alias="Do Zone Sizing Calculation"
    )
    Do_System_Sizing_Calculation: str | bool = Field(
        "No", alias="Do System Sizing Calculation"
    )
    Do_Plant_Sizing_Calculation: str | bool = Field(
        "No", alias="Do Plant Sizing Calculation"
    )
    Run_Simulation_for_Sizing_Periods: str | bool = Field(
        "No", alias="Run Simulation for Sizing Periods"
    )
    Run_Simulation_for_Weather_File_Run_Periods: str | bool = Field(
        "Yes", alias="Run Simulation for Weather File Run Periods"
    )
    Do_HVAC_Sizing_Simulation_for_Sizing_Periods: Optional[str | bool] = Field(
        "Yes", alias="Do HVAC Sizing Simulation for Sizing Periods"
    )
    Maximum_Number_of_HVAC_Sizing_Simulation_Passes: Optional[int] = Field(
        1, alias="Maximum Number of HVAC Sizing Simulation Passes"
    )

    @field_validator(
        "Do_Zone_Sizing_Calculation",
        "Do_System_Sizing_Calculation",
        "Do_Plant_Sizing_Calculation",
        "Run_Simulation_for_Sizing_Periods",
        "Run_Simulation_for_Weather_File_Run_Periods",
        "Do_HVAC_Sizing_Simulation_for_Sizing_Periods",
        mode="before",
    )
    def convert_bool_to_yes_no(cls, v):
        if isinstance(v, bool):
            return "Yes" if v else "No"
        return v


class TimestepSchema(BaseSchema):
    Number_of_Timesteps_per_Hour: int = Field(4, alias="Number of Timesteps per Hour")

    @field_validator("Number_of_Timesteps_per_Hour")
    def validate_timesteps(cls, v):
        if v < 1:
            raise ValueError("Number of Timesteps per Hour must be at least 1.")
        return v


class SiteLocationSchema(BaseSchema):
    Name: str = Field(..., alias="Name")
    Latitude: float = Field(..., alias="Latitude")
    Longitude: float = Field(..., alias="Longitude")
    Time_Zone: float = Field(..., alias="Time Zone")
    Elevation: float = Field(..., alias="Elevation")

    @field_validator("Name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v

    @field_validator("Latitude")
    def validate_latitude(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        return v

    @field_validator("Longitude")
    def validate_longitude(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        return v

    @field_validator("Time_Zone")
    def validate_time_zone(cls, v):
        if not (-12 <= v <= 14):
            raise ValueError("Time Zone must be between -12 and 14 hours.")
        return v


class RunPeriodSchema(BaseSchema):
    Name: str = Field(..., alias="Name")
    Begin_Month: int = Field(..., alias="Begin Month")
    Begin_Day_of_Month: int = Field(..., alias="Begin Day of Month")
    Begin_Year: Optional[int] = Field(None, alias="Begin Year")
    End_Month: int = Field(..., alias="End Month")
    End_Day_of_Month: int = Field(..., alias="End Day of Month")
    End_Year: Optional[int] = Field(None, alias="End Year")
    Day_of_Week_for_Start_Day: Optional[str] = Field(
        None, alias="Day of Week for Start Day"
    )
    Use_Weather_File_Holidays_and_Special_Days: Optional[str | bool] = Field(
        None, alias="Use Weather File Holidays and Special Days"
    )
    Use_Weather_File_Daylight_Saving_Period: Optional[str | bool] = Field(
        None, alias="Use Weather File Daylight Saving Period"
    )
    Apply_Weekend_Holiday_Rule: Optional[str | bool] = Field(
        None, alias="Apply Weekend Holiday Rule"
    )
    Use_Weather_File_Rain_Indicators: Optional[str | bool] = Field(
        None, alias="Use Weather File Rain Indicators"
    )
    Use_Weather_File_Snow_Indicators: Optional[str | bool] = Field(
        None, alias="Use Weather File Snow Indicators"
    )

    @field_validator(
        "Use_Weather_File_Holidays_and_Special_Days",
        "Use_Weather_File_Daylight_Saving_Period",
        "Apply_Weekend_Holiday_Rule",
        "Use_Weather_File_Rain_Indicators",
        "Use_Weather_File_Snow_Indicators",
        mode="before",
    )
    def convert_bool_to_yes_no_runperiod(cls, v):
        if v is None:
            return None
        if isinstance(v, bool):
            return "Yes" if v else "No"
        return v

    @field_validator("Begin_Month", "End_Month")
    def validate_month(cls, v):
        if not (1 <= v <= 12):
            raise ValueError("Month must be between 1 and 12.")
        return v

    @model_validator(mode="after")
    def validate_month_oder(self):
        if self.Begin_Month > self.End_Month:
            raise ValueError("Begin Month must be less than or equal to End Month.")
        return self

    @field_validator("Begin_Day_of_Month", "End_Day_of_Month")
    def validate_day(cls, v):
        if not (1 <= v <= 31):
            raise ValueError("Day of Month must be between 1 and 31.")
        return v

    @model_validator(mode="after")
    def validate_day_order(self):
        if (
            self.Begin_Month == self.End_Month
            and self.Begin_Day_of_Month > self.End_Day_of_Month
        ):
            raise ValueError(
                "Begin Day of Month must be less than or equal to End Day of Month when Begin Month equals End Month."
            )
        return self

    @field_validator("Day_of_Week_for_Start_Day")
    def validate_day_of_week(cls, v):
        valid_days = getattr(cls._idf_field, "RunPeriod").Day_of_Week_for_Start_Day.key
        if v is not None and v not in valid_days:
            raise ValueError(f"Day of Week for Start Day must be one of {valid_days}.")
        return v


class GlobalGeometryRulesSchema(BaseSchema):
    Starting_Vertex_Position: str = Field(..., alias="Starting Vertex Position")
    Vertex_Entry_Direction: str = Field(..., alias="Vertex Entry Direction")
    Coordinate_System: str = Field(..., alias="Coordinate System")

    @field_validator("Starting_Vertex_Position")
    def validate_starting_vertex_position(cls, v):
        valid_positions = getattr(
            cls._idf_field, "GlobalGeometryRules"
        ).Starting_Vertex_Position.key
        if v not in valid_positions:
            raise ValueError(
                f"Starting Vertex Position must be one of {valid_positions}."
            )
        return v

    @field_validator("Vertex_Entry_Direction")
    def validate_vertex_entry_direction(cls, v):
        valid_directions = getattr(
            cls._idf_field, "GlobalGeometryRules"
        ).Vertex_Entry_Direction.key
        return cls.validate_choice_field(v, valid_directions, "Vertex Entry Direction")

    @field_validator("Coordinate_System")
    def validate_coordinate_system(cls, v):
        valid_systems = getattr(
            cls._idf_field, "GlobalGeometryRules"
        ).Coordinate_System.key
        return cls.validate_choice_field(v, valid_systems, "Coordinate System")


class OutputVariableDictionarySchema(BaseSchema):
    Key_Field: str = Field("Regular", alias="Key Field")

    @field_validator("Key_Field")
    def validate_key_field(cls, v):
        valid_key_field = getattr(
            cls._idf_field, "Output_VariableDictionary"
        ).Key_Field.key
        return cls.validate_choice_field(v, valid_key_field, "Key Field")


class OutputDiagnosticsSchema(BaseSchema):
    Key_1: str = Field(..., alias="Key 1")

    @field_validator("Key_1")
    def validate_key_1(cls, v):
        valid_key_1 = getattr(cls._idf_field, "Output_Diagnostics").Key_1.key
        return cls.validate_choice_field(v, valid_key_1, "Key 1")


class OutputTableSummaryReportsSchema(BaseSchema):
    Report_1_Name: str = Field(..., alias="Report 1 Name")

    @field_validator("Report_1_Name")
    def validate_report_1_name(cls, v):
        valid_report_names = getattr(
            cls._idf_field, "Output_Table_SummaryReports"
        ).Report_1_Name.key
        return cls.validate_choice_field(v, valid_report_names, "Report 1 Name")


class OutputControlTableStyleSchema(BaseSchema):
    Column_Separator: str = Field("HTML", alias="Column Separator")
    Unit_Conversion: str = Field("None", alias="Unit Conversion")

    @field_validator("Column_Separator")
    def validate_column_separator(cls, v):
        valid_separators = getattr(
            cls._idf_field, "OutputControl_Table_Style"
        ).Column_Separator.key
        return cls.validate_choice_field(v, valid_separators, "Column Separator")

    @field_validator("Unit_Conversion")
    def validate_unit_conversion(cls, v):
        valid_conversions = getattr(
            cls._idf_field, "OutputControl_Table_Style"
        ).Unit_Conversion.key
        return cls.validate_choice_field(v, valid_conversions, "Unit Conversion")


class OutputVariableSchema(BaseSchema):
    Key_Value: str = Field("*", alias="Key Value")
    Variable_Name: str = Field(..., alias="Variable Name")
    Reporting_Frequency: str = Field("Hourly", alias="Reporting Frequency")

    @field_validator("Reporting_Frequency")
    def validate_reporting_frequency(cls, v):
        valid_frequencies = getattr(
            cls._idf_field, "Output_Variable"
        ).Reporting_Frequency.key
        return cls.validate_choice_field(v, valid_frequencies, "Reporting Frequency")
