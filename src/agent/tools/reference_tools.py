"""Semantic search over DOE prototype objects, for the phases that create them."""

import json
from typing import Any, Final

from idfpy.models.constructions import Material
from langchain_core.tools import BaseTool, tool

from src.reference.extract import Kind
from src.reference.index import Hit
from src.reference.search import ReferenceSearch

ABSORPTANCE_DEFAULTS: Final = {
    field: Material.model_fields[field].default
    for field in ("thermal_absorptance", "solar_absorptance", "visible_absorptance")
}
"""EnergyPlus defaults; a result names an absorptance only when it differs."""

REFERENCE_PROMPT = """
Reference library:
{tools} search materials, constructions and schedules of the DOE
commercial (ASHRAE 90.1-2019/2022) and residential (IECC 2021/2024)
prototype buildings, by meaning. Describe what you need in words, e.g.
"exterior wall insulation for a medium office" or "office occupancy on
weekdays". Results come from the building site's climate zones
({zones}, from the weather file) unless climate_specific is False.
{steps}- Values the specification gives always take precedence, and so do the
  names it gives. A result's name is only a fallback, rewritten with
  letters, digits and '_' only (e.g. 'Window_U_0.504' -> 'Window_U_0p504').
- Prototypes follow US energy codes; they are reference values, not proof
  of compliance with local standards.
"""


def _material_values(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in data.items()
        if ABSORPTANCE_DEFAULTS.get(key, object()) != value
    }


def compact_hit(hit: Hit, *, with_material_values: bool) -> dict[str, Any]:
    """A result as the phase needs it: every later call resends it.

    Kept: what the phase uses to choose (score, name, roles, building types,
    climate zones, standards) and the values it creates from. Left out: the
    kind and the name again inside the data, a construction's object type,
    and absorptances at the EnergyPlus default. A construction carries its
    material values only for a phase that creates materials; the
    construction phase names existing materials.
    """
    data = {k: v for k, v in hit.data.items() if k != "name"}
    if hit.kind == "material":
        data = _material_values(data)
    elif "materials" in data:
        if with_material_values:
            data["materials"] = [_material_values(m) for m in data["materials"]]
        else:
            del data["materials"]
    entry: dict[str, Any] = {"score": round(hit.score, 2), "name": hit.name}
    if hit.kind == "material":
        entry["object_type"] = hit.object_type
    return entry | {
        "data": data,
        "roles": hit.roles,
        "building_types": hit.building_types,
        "climate_zones": hit.climate_zones,
        "standards": hit.standards,
    }


def _search_tool(
    search: ReferenceSearch,
    name: str,
    kind: Kind,
    description: str,
    *,
    with_material_values: bool,
) -> BaseTool:
    def find(query: str, climate_specific: bool = True, limit: int = 5) -> str:
        hits = search.find(
            query, kind, by_climate=climate_specific, limit=min(max(limit, 1), 10)
        )
        entries = [
            compact_hit(h, with_material_values=with_material_values) for h in hits
        ]
        return json.dumps(entries, separators=(",", ":"), default=str)

    find.__name__ = name
    find.__doc__ = description
    return tool(find)


def make_reference_tools(
    search: ReferenceSearch, kinds: tuple[Kind, ...]
) -> list[BaseTool]:
    """Search tools for the given kinds of reference object.

    Construction results carry their material values only when ``kinds``
    includes materials, i.e. for the phase that creates them.
    """
    with_material_values = "material" in kinds
    layers = (
        "`layers` (outside to inside) and the `materials` of those layers, "
        "with absorptances only where they differ from the EnergyPlus defaults"
        if with_material_values
        else "`layers` (outside to inside): the names of the materials to use"
    )
    descriptions: dict[Kind, str] = {
        "material": """Find prototype materials similar to a description.

        Each result's `data` holds the material's fields, as in the lists
        of create_materials (WindowMaterial:Glazing calls solar
        transmittance `solar_transmittance_at_normal_incidence`; a glass
        pane's `solar_transmittance`). Absorptances appear only where they
        differ from the EnergyPlus defaults; pass them on when they do.

        Args:
            query: What the material is for, in words.
            climate_specific: Only materials used in the site's climate zones.
            limit: Number of results, 1-10.
        """,
        "construction": f"""Find prototype constructions similar to a description.

        Each result's `data` has {layers}.

        Args:
            query: The element and building, in words.
            climate_specific: Only constructions used in the site's zones.
            limit: Number of results, 1-10.
        """,
        "schedule": """Find prototype schedules similar to a description.

        Each result's `data` has `periods` in the create_schedule_compact
        `data` format and its `schedule_type_limits`.

        Args:
            query: What the schedule controls and when, in words.
            climate_specific: Only schedules of the site's climate zones;
                usually False, as schedules rarely depend on climate.
            limit: Number of results, 1-10.
        """,
    }
    return [
        _search_tool(
            search,
            f"find_reference_{kind}s",
            kind,
            descriptions[kind],
            with_material_values=with_material_values,
        )
        for kind in kinds
    ]


# What each kind of search is required for: a phase left to decide whether
# to search mostly did not, and invented values instead.
_STEPS: dict[Kind, str] = {
    "construction": """- REQUIRED for every envelope construction the specification gives no
  layer values for (exterior walls, roofs, ground floors, windows): call
  `find_reference_constructions` with the element and building type,
  keeping climate_specific True, BEFORE creating anything for it. Use the
  best fitting result's layers in its order; a phase that creates
  materials takes exactly their values, absorptances included.
""",
    "material": """- Any other material without values in the specification: call
  `find_reference_materials` and take the values of a fitting result
  exactly, absorptances included, rather than typical values from memory.
""",
    "schedule": """- REQUIRED for every schedule the specification gives no profile for:
  call `find_reference_schedules` (climate_specific False) with what it
  controls and the building type, and use the best result's periods and
  type limits.
""",
}


def reference_prompt(search: ReferenceSearch, kinds: tuple[Kind, ...]) -> str:
    tools = " and ".join(f"`find_reference_{kind}s`" for kind in kinds)
    zones = ", ".join(search.climate.candidates)
    steps = "".join(_STEPS[kind] for kind in sorted(kinds))
    return REFERENCE_PROMPT.format(tools=tools, zones=zones, steps=steps)
