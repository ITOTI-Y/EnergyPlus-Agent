import pytest
from idfpy import IDF
from idfpy.models.constructions import Construction, Material, MaterialNoMass
from idfpy.models.schedules import (
    ScheduleDayHourly,
    ScheduleWeekCompact,
    ScheduleWeekCompactDataItem,
    ScheduleYear,
    ScheduleYearScheduleWeeksItem,
)

from src.modeling.schedules import ThroughSchema
from src.reference.extract import compact_periods, extract_objects, year_periods


def test_compact_fields_split_combined_day_types_and_keep_their_values():
    periods = compact_periods(
        [
            "Through:12/31",
            "For: Weekdays SummerDesignDay",
            "Until: 08:00",
            "0.1",
            "Until: 24:00, 0.9",
            "For:Saturdays Holiday AllOtherDays",
            "Until: 24:00",
            "0",
        ]
    )

    days = [
        (d["For"], [t["Until"]["Value"] for t in d["Times"]])
        for d in periods[0]["Days"]
    ]
    assert days == [
        ("Weekdays", [0.1, 0.9]),
        ("SummerDesignDay", [0.1, 0.9]),
        ("Saturday", [0.0]),
        ("Holidays", [0.0]),
        ("AllOtherDays", [0.0]),
    ]
    ThroughSchema.model_validate(periods[0])


def test_compact_fields_with_interpolation_are_refused():
    with pytest.raises(ValueError, match="interpolation"):
        compact_periods(["Through: 12/31", "For: AllDays", "Interpolate: Average"])


def test_year_schedule_becomes_periods_with_equal_hours_merged():
    idf = IDF()
    idf.add(
        ScheduleDayHourly.model_validate(
            {"name": "Flat"} | {f"hour_{h}": 0.5 for h in range(1, 25)}
        )
    )
    idf.add(
        ScheduleDayHourly.model_validate(
            {"name": "Day"}
            | {f"hour_{h}": 1.0 if 9 <= h <= 17 else 0.0 for h in range(1, 25)}
        )
    )
    idf.add(
        ScheduleWeekCompact(
            name="Week",
            data=[
                ScheduleWeekCompactDataItem(
                    daytype_list="For: Weekdays", schedule_day_name="Day"
                ),
                ScheduleWeekCompactDataItem(
                    daytype_list="AllOtherDays", schedule_day_name="Flat"
                ),
            ],
        )
    )
    year = ScheduleYear(
        name="Year",
        schedule_weeks=[
            ScheduleYearScheduleWeeksItem(
                schedule_week_name="Week",
                start_month=1,
                start_day=1,
                end_month=12,
                end_day=31,
            )
        ],
    )
    idf.add(year)

    [period] = year_periods(idf, year)

    assert period["Through"] == "12/31"
    weekdays, others = period["Days"]
    assert [(t["Until"]["Time"], t["Until"]["Value"]) for t in weekdays["Times"]] == [
        ("08:00", 0.0),
        ("17:00", 1.0),
        ("24:00", 0.0),
    ]
    assert others["Times"] == [{"Until": {"Time": "24:00", "Value": 0.5}}]


def test_construction_carries_its_layer_materials_and_uses():
    idf = IDF()
    idf.add(
        Material(
            name="Concrete",
            roughness="Rough",
            thickness=0.2,
            conductivity=1.4,
            density=2240.0,
            specific_heat=900.0,
        )
    )
    idf.add(
        MaterialNoMass(name="Insulation", roughness="Smooth", thermal_resistance=2.0)
    )
    idf.add(
        Construction(name="Ext Wall", outside_layer="Concrete", layer_2="Insulation")
    )
    idf.add(MaterialNoMass(name="Spare", roughness="Smooth", thermal_resistance=1.0))
    idf.add(Construction(name="Unused Wall", outside_layer="Spare"))

    objects = extract_objects(idf, {"ext wall": {"Wall, Outdoors"}}).objects

    [wall] = [o for o in objects if o.kind == "construction"]
    assert wall.data["layers"] == ["Concrete", "Insulation"]
    assert [m["object_type"] for m in wall.data["materials"]] == [
        "Material",
        "Material:NoMass",
    ]
    assert wall.roles == {"Wall, Outdoors"}
    insulation = next(o for o in objects if o.name == "Insulation")
    assert insulation.roles == {"Wall, Outdoors"}
    # On no surface: left out, with the material only it uses.
    assert {"Unused Wall", "Spare"}.isdisjoint(o.name for o in objects)
