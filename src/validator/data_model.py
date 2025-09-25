from pydantic import BaseModel, ConfigDict, Field, field_validator

class PydanticConfig:
    model_config = ConfigDict(
        from_attributes=True,  # 支持从对象创建模型
        validate_assignment=True,  # 赋值时验证
        arbitrary_types_allowed=True,  # 允许任意类型
        str_strip_whitespace=True,  # 自动去除字符串空格
        use_enum_values=True,  # 使用枚举值
    )

class BuildingSchema(BaseModel, PydanticConfig):
    name: str = Field(..., alias="Name", description="Building name")
    

    @field_validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v
    
class ZoneSchema(BaseModel, PydanticConfig):
    name: str = Field(..., description="Zone name")
    
    @field_validator('name')
    def validate_name(cls, v):
        if not v:
            raise ValueError("Name must not be empty.")
        return v