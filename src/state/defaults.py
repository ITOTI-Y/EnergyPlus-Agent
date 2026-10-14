"""Objects every simulation model starts with, and design days from a .ddy file."""

from pathlib import Path
from typing import Final

from idfpy import IDF
from idfpy.models import (
    GlobalGeometryRules,
    OutputControlTableStyle,
    OutputDiagnostics,
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReports,
    OutputTableSummaryReportsReportsItem,
    OutputVariable,
    OutputVariableDictionary,
    RunPeriod,
    SimulationControl,
    SizingPeriodDesignDay,
    Timestep,
    Version,
)

# ASHRAE annual heating 99.6% and cooling 0.4% dry-bulb conditions; a .ddy file
# prefixes each name with the station, e.g. "Shenzhen Ann Htg 99.6% Condns DB".
_DESIGN_DAY_SUFFIXES: Final = (
    " Ann Htg 99.6% Condns DB",
    " Ann Clg .4% Condns DB=>MWB",
)

# Without at least one Output:Variable, EnergyPlus runs the full RunPeriod
# but `eplusout.eso` stays 0 bytes.
_DEFAULT_OUTPUT_VARIABLES: Final = (
    ("*", "Zone Mean Air Temperature"),
    ("*", "Zone Air Relative Humidity"),
    ("*", "Zone Ideal Loads Supply Air Total Heating Energy"),
    ("*", "Zone Ideal Loads Supply Air Total Cooling Energy"),
    ("*", "Zone Lights Electricity Energy"),
    ("*", "Zone Electric Equipment Electricity Energy"),
    ("*", "Zone People Total Heating Energy"),
    ("", "Facility Total HVAC Electricity Demand Rate"),
)


def new_model() -> IDF:
    """Create an IDF with the global objects a simulation needs."""
    idf = IDF()
    for obj in (
        Version(),
        SimulationControl(),
        Timestep(),
        GlobalGeometryRules(
            starting_vertex_position="UpperLeftCorner",
            vertex_entry_direction="Counterclockwise",
            coordinate_system="World",
        ),
        RunPeriod(
            name="Annual",
            begin_month=1,
            begin_day_of_month=1,
            end_month=12,
            end_day_of_month=31,
        ),
        OutputVariableDictionary(),
        OutputDiagnostics(
            diagnostics=[OutputDiagnosticsDiagnosticsItem(key="DisplayExtraWarnings")]
        ),
        OutputTableSummaryReports(
            reports=[OutputTableSummaryReportsReportsItem(report_name="AllSummary")]
        ),
        OutputControlTableStyle(),
    ):
        idf.add(obj)
    return idf


def add_design_days(idf: IDF, ddy_path: Path) -> None:
    """Add the annual heating and cooling design days from a .ddy file.

    Raises:
        FileNotFoundError: If ``ddy_path`` does not exist.
        ValueError: If the file holds an object idfpy cannot validate.
        LookupError: If the file lacks one of the design days.
    """
    ddy = IDF.load(ddy_path, strict=True)
    available = ddy.all_of_type(SizingPeriodDesignDay)
    names: list[str] = []
    for suffix in _DESIGN_DAY_SUFFIXES:
        name = next((n for n in available if n.endswith(suffix)), None)
        if name is None:
            raise LookupError(f"{ddy_path} has no design day ending with {suffix!r}")
        names.append(name)
    for name in names:
        ddy.remove(SizingPeriodDesignDay, name)
        idf.add(available[name])


def add_output_variables(idf: IDF) -> None:
    for key, name in _DEFAULT_OUTPUT_VARIABLES:
        idf.add(
            OutputVariable(
                key_value=key,
                variable_name=name,
                reporting_frequency="Hourly",
            )
        )
