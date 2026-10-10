"""Objects every simulation model starts with, and design days from a .ddy file."""

from pathlib import Path
from typing import Final

from idfpy import IDF
from idfpy.models.location import RunPeriod, SizingPeriodDesignDay
from idfpy.models.outputs import (
    OutputControlTableStyle,
    OutputDiagnostics,
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReports,
    OutputTableSummaryReportsReportsItem,
    OutputVariableDictionary,
)
from idfpy.models.simulation import SimulationControl, Timestep, Version
from idfpy.models.thermal_zones import GlobalGeometryRules

# ASHRAE annual heating 99.6% and cooling 0.4% dry-bulb conditions; a .ddy file
# prefixes each name with the station, e.g. "Shenzhen Ann Htg 99.6% Condns DB".
DESIGN_DAY_SUFFIXES: Final = (
    " Ann Htg 99.6% Condns DB",
    " Ann Clg .4% Condns DB=>MWB",
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
        LookupError: If the file lacks one of the design days.
    """
    ddy = IDF.load(ddy_path)
    available = ddy.all_of_type(SizingPeriodDesignDay)
    for suffix in DESIGN_DAY_SUFFIXES:
        name = next((n for n in available if n.endswith(suffix)), None)
        if name is None:
            raise LookupError(f"{ddy_path} has no design day ending with {suffix!r}")
        # remove() unbinds the object from the .ddy model so idf can own it.
        ddy.remove(SizingPeriodDesignDay, name)
        idf.add(available[name])
