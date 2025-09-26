from typing import Tuple, List
from pydantic import BaseModel, ConfigDict, Field, field_validator

class PydanticConfig:
    model_config = ConfigDict(
        from_attributes=True,  # 支持从对象创建模型
        validate_assignment=True,  # 赋值时验证
        arbitrary_types_allowed=True,  # 允许任意类型
        str_strip_whitespace=True,  # 自动去除字符串空格
        use_enum_values=True,  # 使用枚举值
        populate_by_name=True,  # 允许通过字段名填充
        extra='ignore'  # 忽略额外字段
    )

class BuildingSchema(BaseModel, PydanticConfig):
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

class SettingsSchema(BaseModel, PydanticConfig):
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

class ZoneSchema(BaseModel, PydanticConfig):
    name: str = Field(..., alias="Name", description="Zone name")
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v