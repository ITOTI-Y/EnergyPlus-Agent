"""EnergyPlus simulation results – parsing and visualisation.

Public API
----------
from src.results import load_results, SimulationResult
from src.results import parse_idf_geometry, ZoneGeometry
from src.results import charts
"""

from src.results.idf_geometry import (
    FenestrationPolygon,
    SurfacePolygon,
    ZoneGeometry,
    csv_key_to_idf_zone,
    idf_zone_to_csv_key,
    parse_fenestrations,
    parse_idf_geometry,
)
from src.results.parser import (
    SimulationResult,
    load_results,
    parse_tabular,
    parse_timeseries,
)

__all__ = [
    # parser
    "SimulationResult",
    "load_results",
    "parse_timeseries",
    "parse_tabular",
    # idf geometry
    "ZoneGeometry",
    "SurfacePolygon",
    "FenestrationPolygon",
    "parse_idf_geometry",
    "parse_fenestrations",
    "idf_zone_to_csv_key",
    "csv_key_to_idf_zone",
]
