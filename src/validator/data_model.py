from dateutil.parser import parse
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ScheduleCompactSchema(BaseModel):
    """Validates nested Through/For/Until input and flattens it into fields."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    name: str = Field(..., alias="Name")
    schedule_type_limits_name: str = Field(..., alias="Schedule Type Limits Name")
    data: list = Field(..., alias="Data", min_length=1)

    @field_validator("name", "schedule_type_limits_name")
    def validate_non_empty(cls, v: str, info: "ValidationInfo") -> str:
        if not v:
            raise ValueError(f"Field '{info.field_name}' must not be empty.")
        return v

    @field_validator("data")
    def validate_data(cls, v: list) -> list[str]:
        if v and all(isinstance(x, str) for x in v):
            return v
        return cls._validate_through(v)

    @classmethod
    def _validate_through(cls, data: list) -> list[str]:
        result = []
        for i, item in enumerate(data):
            date = item["Through"]
            date = parse(date).strftime("%m/%d")
            day_data = cls._validate_for(item["Days"])
            if i == len(data) - 1 and date != "12/31":
                raise ValueError("Schedule data must end with Through: 12/31")
            result.append(f"Through: {date}")
            result.extend(day_data)
        return result

    @classmethod
    def _validate_for(cls, data: list) -> list[str]:
        VALID_DAY_TYPES = {
            "weekdays",
            "weekends",
            "holidays",
            "alldays",
            "summerdesignday",
            "winterdesignday",
            "sunday",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "customday1",
            "customday2",
            "allotherdays",
        }
        result = []
        for _, item in enumerate(data):
            day_type = item["For"]
            if day_type.lower() not in VALID_DAY_TYPES:
                raise ValueError(f"Invalid day type: {day_type}")
            time_data = cls._validate_until(item["Times"])
            result.append(f"For: {day_type}")
            result.extend(time_data)
        return result

    @classmethod
    def _validate_until(cls, data: list) -> list[str]:
        result = []
        for i, item in enumerate(data):
            time = item["Until"]["Time"]
            value = float(item["Until"]["Value"])
            if i == len(data) - 1:
                if time != "24:00":
                    raise ValueError(f"Last time entry must be 24:00, but got {time}")
            else:
                time = parse(time).strftime("%H:%M")
            # Separate fields: epJSON keeps "Until: 24:00, 1.0" as one value,
            # which EnergyPlus cannot read.
            result.extend((f"Until: {time}", str(value)))
        return result
