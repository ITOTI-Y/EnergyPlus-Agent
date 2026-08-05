import pytest

from src.validator.data_model import ScheduleTypeLimitsSchema


def test_schedule_type_limits_none_limits_normalized_to_empty_string():
    schema = ScheduleTypeLimitsSchema.model_validate(
        {"Name": "Any Number", "Lower Limit Value": None, "Upper Limit Value": None}
    )

    assert schema.lower_limit_value == ""
    assert schema.upper_limit_value == ""


def test_schedule_type_limits_numeric_limits_accepted():
    schema = ScheduleTypeLimitsSchema.model_validate(
        {"Name": "Fraction", "Lower Limit Value": 0.0, "Upper Limit Value": 1.0}
    )

    assert schema.lower_limit_value == 0.0
    assert schema.upper_limit_value == 1.0


def test_schedule_type_limits_rejects_non_numeric_string():
    with pytest.raises(ValueError, match="number or an empty string"):
        ScheduleTypeLimitsSchema.model_validate(
            {"Name": "Bad", "Lower Limit Value": "low", "Upper Limit Value": 1.0}
        )
