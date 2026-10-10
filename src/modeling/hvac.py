"""HVACTemplate ideal loads systems, which have no name and are keyed by zone."""

from idfpy import IDF
from idfpy.models.hvac_templates import HVACTemplateZoneIdealLoadsAirSystem

from src.modeling.errors import DuplicateNameError, ObjectNotFoundError

_TYPE = HVACTemplateZoneIdealLoadsAirSystem.idf_object_type()


def _systems_of(
    idf: IDF, zone_name: str
) -> list[tuple[str, HVACTemplateZoneIdealLoadsAirSystem]]:
    return [
        (key, system)
        for key, system in idf.all_of_type(HVACTemplateZoneIdealLoadsAirSystem).items()
        if system.zone_name == zone_name
    ]


def find_ideal_loads(
    idf: IDF, zone_name: str
) -> tuple[str, HVACTemplateZoneIdealLoadsAirSystem]:
    """Storage key and system of the zone.

    Raises:
        ObjectNotFoundError: If the zone has no ideal loads system.
    """
    if not (found := _systems_of(idf, zone_name)):
        raise ObjectNotFoundError(_TYPE, zone_name)
    return found[0]


def check_zone_free(idf: IDF, zone_name: str) -> None:
    """Raises: DuplicateNameError: If the zone already has a system."""
    if _systems_of(idf, zone_name):
        raise DuplicateNameError(_TYPE, zone_name)
