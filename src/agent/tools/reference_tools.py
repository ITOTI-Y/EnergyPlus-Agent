"""Semantic search over DOE prototype objects, for the phases that create them."""

import json
from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from src.reference.extract import Kind
from src.reference.search import ReferenceSearch

REFERENCE_PROMPT = """
Reference library:
{tools} search materials, constructions and schedules of the DOE
commercial (ASHRAE 90.1-2019/2022) and residential (IECC 2021/2024)
prototype buildings, by meaning. Describe what you need in words, e.g.
"exterior wall insulation for a medium office" or "office occupancy on
weekdays". Results come from the building site's climate zones
({zones}, from the weather file) unless climate_specific is False.
- Search before inventing properties the specification does not give, and
  take the values of a fitting result.
- Values the specification gives always take precedence, and so do the
  names it gives. A result's name is only a fallback, rewritten with
  letters, digits and '_' only (e.g. 'Window_U_0.504' -> 'Window_U_0p504').
- Prototypes follow US energy codes; they are reference values, not proof
  of compliance with local standards.
"""


def _search_tool(
    search: ReferenceSearch, name: str, kind: Kind, description: str
) -> BaseTool:
    def find(query: str, climate_specific: bool = True, limit: int = 5) -> str:
        hits = search.find(
            query, kind, by_climate=climate_specific, limit=min(max(limit, 1), 10)
        )
        return json.dumps([asdict(h) for h in hits], default=str)

    find.__name__ = name
    find.__doc__ = description
    return tool(find)


def make_reference_tools(
    search: ReferenceSearch, kinds: tuple[Kind, ...]
) -> list[BaseTool]:
    """Search tools for the given kinds of reference object."""
    descriptions: dict[Kind, str] = {
        "material": """Find prototype materials similar to a description.

        Each result's `data` holds the fields of create_* material tools
        (WindowMaterial:Glazing calls solar transmittance
        `solar_transmittance_at_normal_incidence`).

        Args:
            query: What the material is for, in words.
            climate_specific: Only materials used in the site's climate zones.
            limit: Number of results, 1-10.
        """,
        "construction": """Find prototype constructions similar to a description.

        Each result's `data` has `layers` (outside to inside) and the full
        `materials` of those layers.

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
        _search_tool(search, f"find_reference_{kind}s", kind, descriptions[kind])
        for kind in kinds
    ]


def reference_prompt(search: ReferenceSearch, kinds: tuple[Kind, ...]) -> str:
    tools = " and ".join(f"`find_reference_{kind}s`" for kind in kinds)
    zones = ", ".join(search.climate.candidates)
    return REFERENCE_PROMPT.format(tools=tools, zones=zones)
