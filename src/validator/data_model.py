from typing import Tuple, List, Optional, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class IDDField:
    def __init__(self, data: List[Dict] | Dict):
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, list):
                    if len(obj) > 0 and isinstance(obj[0], dict):
                        obj_name = obj[0].get('idfobj', None)
                    else:
                        continue
                    if obj_name:
                        obj_name = self._clean_key(obj_name)
                        setattr(self, obj_name, IDDField(obj[1:]))
                elif isinstance(obj, dict):
                    field_name = obj.get('field', None)
                    if field_name:
                        if isinstance(field_name, (list, tuple)) and len(field_name) > 0:
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
        extra='ignore'  # 忽略额外字段
    )

    _idf_field: IDDField = IDDField({})

    @classmethod
    def set_idf_field(cls, idf_field: IDDField):
        cls._idf_field = idf_field

    @property
    def idf_field(self) -> IDDField:
        return self._idf_field

class BuildingSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Building name")
    north_axis: float = Field(0.0, alias="North Axis", description="Building north axis in degrees")
    terrain: str = Field("Suburbs", alias="Terrain", description="Terrain type")
    loads_convergence_tolerance_value: float = Field(0.04, alias="Loads Convergence Tolerance Value", description="Loads convergence tolerance value")
    temperature_convergence_tolerance_value: float = Field(0.4, alias="Temperature Convergence Tolerance Value", description="Temperature convergence tolerance value")
    solar_distribution: str = Field("FullExterior", alias="Solar Distribution", description="Solar distribution")
    maximum_number_of_warmup_days: int = Field(25, alias="Maximum Number of Warmup Days", description="Maximum number of warmup days")
    minimum_number_of_warmup_days: int = Field(0, alias="Minimum Number of Warmup Days", description="Minimum number of warmup days")

    @field_validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v
    @field_validator('north_axis')
    def validate_north_axis(cls, v):
        if not (0 <= v < 360):
            raise ValueError("North Axis must be in [0, 360).")
        return v
    @field_validator('terrain')
    def validate_terrain(cls, v):
        valid_terrains = {"Suburbs", "Country", "City", "Ocean", "Urban"}
        if v not in valid_terrains:
            raise ValueError(f"Terrain must be one of {valid_terrains}.")
        return v
    @field_validator('loads_convergence_tolerance_value', 'temperature_convergence_tolerance_value')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive.")
        return v
    @field_validator('solar_distribution')
    def validate_solar_distribution(cls, v):
        valid_distribution = {"FullExterior", "MinimalShadowing", "FullInteriorAndExterior", "FullExteriorWithReflections", "FullInteriorAndExteriorWithReflections"}
        if v not in valid_distribution:
            raise ValueError(f"Solar Distribution must be one of {valid_distribution}.")
        return v
    @field_validator('maximum_number_of_warmup_days', 'minimum_number_of_warmup_days')
    def validate_warmup_days(cls, v):
        if v < 0:
            raise ValueError("Warmup days must be non-negative.")
        return v

class SettingsSchema(BaseSchema):
    version: str | Tuple | List  = Field(..., alias="Version Identifier", description="Version identifier")
    
    @field_validator('version')
    def validate_version(cls, v):
        if not v:
            raise ValueError("Version Identifier must not be empty.")
        if isinstance(v, (list, tuple)):
            return ".".join([str(i) for i in v])
        if isinstance(v, str):
            return v
        raise ValueError("Version Identifier must be a string or a tuple/list of integers.")

class ZoneSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Zone name")
    direction_of_relative_north: Optional[float] = Field(0.0, alias="Direction of Relative North", description="Direction of relative north in degrees")
    x_origin: float = Field(0.0, alias="X Origin", description="X origin coordinate")
    y_origin: float = Field(0.0, alias="Y Origin", description="Y origin coordinate")
    z_origin: float = Field(0.0, alias="Z Origin", description="Z origin coordinate")
    type: int = Field(1, alias="Type", description="Zone type is currently unused in EnergyPlus")
    multiplier: int = Field(1, alias="Multiplier", description="Zone multiplier", ge=1)
    ceiling_height: str | float = Field("autocalculate", alias="Ceiling Height", description="Ceiling height in meters or 'autocalculate'")
    volume: str | float = Field("autocalculate", alias="Volume", description="Zone volume in cubic meters or 'autocalculate'")
    floor_area: str | float = Field("autocalculate", alias="Floor Area", description="Zone floor area in square meters or 'autocalculate'")
    zone_inside_convection_algorithm: str = Field("TARP", alias="Zone Inside Convection Algorithm", description="Zone inside convection algorithm")
    zone_outside_convection_algorithm: str = Field("DOE-2", alias="Zone Outside Convection Algorithm", description="Zone outside convection algorithm")
    part_of_total_floor_area: str = Field("Yes", alias="Part of Total Floor Area", description="Part of total floor area")
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v
    @field_validator('direction_of_relative_north')
    def validate_direction_of_relative_north(cls, v):
        if v is not None and not (0 <= v < 360):
            raise ValueError("Direction of Relative North must be in [0, 360).")
        elif v is None:
            return 0.0
        return v
    @field_validator('x_origin', 'y_origin', 'z_origin')
    def validate_origin(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError("Origin coordinates must be numeric.")
        return v
    @field_validator('type')
    def validate_type(cls, v):
        if v < 0:
            raise ValueError("Zone Type must be non-negative.")
        return v
    @field_validator('multiplier')
    def validate_multiplier(cls, v):
        if v < 1:
            raise ValueError("Multiplier must be at least 1.")
        return v
    @field_validator('ceiling_height', 'volume', 'floor_area')
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
    @field_validator('zone_inside_convection_algorithm')
    def validate_zone_inside_convection_algorithm(cls, v):
        valid_algorithms = {"Simple", "TARP", "CeilingDiffuser", "AdaptiveConvectionAlgorithm", "TrombeWall", "ASTMC1340"}
        if v not in valid_algorithms:
            raise ValueError(f"Zone Inside Convection Algorithm must be one of {valid_algorithms}.")
        return v
    @field_validator('zone_outside_convection_algorithm')
    def validate_zone_outside_convection_algorithm(cls, v):
        valid_algorithms = {"Simple", "TARP", "DOE-2", "MoWiTT", "AdaptiveConvectionAlgorithm"}
        if v not in valid_algorithms:
            raise ValueError(f"Zone Outside Convection Algorithm must be one of {valid_algorithms}.")
        return v
    @field_validator('part_of_total_floor_area')
    def validate_part_of_total_floor_area(cls, v):
        valid_options = {"Yes", "No"}
        if v not in valid_options:
            raise ValueError(f"Part of Total Floor Area must be one of {valid_options}.")
        return v

class SurfaceSchema(BaseSchema):
    name: str = Field(..., alias="Name", description="Surface name")
    surface_type: str = Field(..., alias="Surface Type", description="Type of surface")
    construction_name: str = Field(..., alias="Construction Name", description="Name of the construction")
    zone_name: str = Field(..., alias="Zone Name", description="Name of the associated zone")
    space_name: Optional[str] = Field(None, alias="Space Name", description="Name of the associated space")
    outside_boundary_condition: str = Field(..., alias="Outside Boundary Condition", description="Outside boundary condition")
    outside_boundary_condition_object: Optional[str] = Field(None, alias="Outside Boundary Condition Object", description="Outside boundary condition object")
    sun_exposure: str = Field("NoSun", alias="Sun Exposure", description="Sun exposure")
    wind_exposure: str = Field("NoWind", alias="Wind Exposure", description="Wind exposure")
    view_factor_to_ground: str | float = Field("autocalculate", alias="View Factor to Ground", description="View factor to ground or 'autocalculate'")
    vertices: List[dict[str, float]] = Field(..., alias="Vertices", description="List of vertices defining the surface")


    @field_validator('name', 'construction_name', 'zone_name')
    def validate_non_empty(cls, v):
        if not v:
            raise ValueError("This field must not be empty.")
        return v
    @field_validator('surface_type')
    def validate_surface_type(cls, v):
        valid_types = getattr(cls._idf_field, 'BuildingSurface_Detailed').Surface_Type.key
        if v not in valid_types:
            raise ValueError(f"Surface Type must be one of {valid_types}.")
        return v

    @field_validator('outside_boundary_condition')
    def validate_outside_boundary_condition(cls, v):
        valid_conditions = getattr(cls._idf_field, 'BuildingSurface_Detailed').Outside_Boundary_Condition.key
        if v not in valid_conditions:
            raise ValueError(f"Outside Boundary Condition must be one of {valid_conditions}.")
        return v
    @field_validator('sun_exposure')
    def validate_sun_exposure(cls, v):
        valid_exposures = getattr(cls._idf_field, 'BuildingSurface_Detailed').Sun_Exposure.key
        if v not in valid_exposures:
            raise ValueError(f"Sun Exposure must be one of {valid_exposures}.")
        return v
    @field_validator('wind_exposure')
    def validate_wind_exposure(cls, v):
        valid_exposures = getattr(cls._idf_field, 'BuildingSurface_Detailed').Wind_Exposure.key
        if v not in valid_exposures:
            raise ValueError(f"Wind Exposure must be one of {valid_exposures}.")
        return v

    @field_validator('view_factor_to_ground')
    def validate_view_factor_to_ground(cls, v):
        if isinstance(v, str) and v.lower() == "autocalculate":
            return "autocalculate"
        try:
            fv = float(v)
            if not (0.0 <= fv <= 1.0):
                raise ValueError("View Factor to Ground must be between 0.0 and 1.0.")
            return fv
        except (TypeError, ValueError):
            raise ValueError("View Factor to Ground must be a number between 0.0 and 1.0 or 'autocalculate'.")


    @field_validator('vertices')
    def validate_vertices(cls, v):
        if not isinstance(v, list) or len(v) < 3:
            raise ValueError("Vertices must be a list with at least three vertex dictionaries.")
        for vertex in v:
            if not (isinstance(vertex, dict) and all(k in vertex for k in ('X', 'Y', 'Z')) and all(isinstance(vertex[k], (int, float)) for k in ('X', 'Y', 'Z'))):
                raise ValueError("Each vertex must be a dictionary with numeric keys 'X', 'Y', and 'Z'.")
        #还需要添加一个验证，确保顶点是按顺序排列的，且形成一个闭合多边形
        
        return v

    @model_validator(mode='after')
    def validate_boundary_condition_object(self):
        needs_obj = {"Surface", "OtherSideCoefficients", "OtherSideConditionsModel"}
        if self.outside_boundary_condition in needs_obj:
            if not self.outside_boundary_condition_object:
                raise ValueError(
                    f"Outside Boundary Condition Object is required when "
                    f"Outside Boundary Condition is '{self.outside_boundary_condition}'."
                )
        return self
