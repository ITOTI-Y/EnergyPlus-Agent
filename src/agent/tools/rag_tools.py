"""RAG reference lookup tools for EnergyPlus phase agents.

RAGSystem is initialized once at first call via a lazy singleton (_get_rag).
If initialization fails (missing env vars, unreachable Qdrant/Gemini), _rag is
set to None and all RAG tools return a graceful "unavailable" response so the
rest of the agent pipeline is unaffected.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv
from langchain_core.tools import BaseTool, tool

from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.rag.rag import RAGSystem

load_dotenv()
_logger = get_logger(__name__)

# ── Lazy singleton ────────────────────────────────────────────────────────────

_rag: RAGSystem | None = None
_rag_attempted: bool = False


def _get_rag() -> RAGSystem | None:
    """Return the shared RAGSystem instance, initializing it on first call.

    Returns None (without raising) if env vars are missing or init fails.
    The _rag_attempted flag ensures a failed init is not retried on every
    node invocation inside a hot LangGraph parallel branch.
    """
    global _rag, _rag_attempted
    if _rag_attempted:
        return _rag
    _rag_attempted = True

    qdrant_url = os.getenv("QDRANT_ENDPOINT", "")
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", "energyplus_database")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if not (qdrant_url and gemini_api_key):
        _logger.warning(
            "RAG unavailable: QDRANT_ENDPOINT or GEMINI_API_KEY is not set. "
            "Phase agents will use ASHRAE default values."
        )
        return None

    try:
        from src.rag.rag import RAGSystem  # local import to keep startup fast

        _rag = RAGSystem(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_collection_name=qdrant_collection,
            gemini_api_key=gemini_api_key,
        )
        _logger.info("RAGSystem initialized (collection={})", qdrant_collection)
    except Exception as exc:
        _logger.warning("RAGSystem initialization failed: {}. RAG tools disabled.", exc)
    return _rag


# ── Table name constants ──────────────────────────────────────────────────────

TABLE_STANDARD_MATERIALS: Final = "standard_materials"
TABLE_NO_MASS_MATERIALS: Final = "no_mass_materials"
TABLE_ALL_MATERIALS: Final = "all_materials"
TABLE_CONSTRUCTIONS: Final = "constructions"
TABLE_SCHEDULE_TYPE_LIMITS: Final = "schedule_type_limits"
TABLE_SCHEDULE_COMPACT: Final = "schedule_compact"
TABLE_SIZING_PERIOD_DESIGN_DAY: Final = "sizingperiod_designday"


# ── Tool factory ──────────────────────────────────────────────────────────────


def make_rag_tool(
    allowed_tables: list[str] | None = None,
    top_k: int = 5,
    score_threshold: float = 0.5,
    rag: RAGSystem | None = None,
) -> BaseTool:
    """Build a ``search_energyplus_reference`` tool pre-scoped to *allowed_tables*.

    Args:
        allowed_tables: Tables to search. Searches are merged and re-ranked by
            cosine score. Pass None to search across all tables.
        top_k: Maximum number of results to return.
        score_threshold: Minimum cosine similarity (0.0-1.0).
        rag: RAGSystem instance to use. If None, the module-level singleton is
            used (obtained via _get_rag()). Pass an explicit instance for testing.
    """
    if rag is None:
        rag = _get_rag()  # lazy singleton — may return None if unavailable

    @tool
    def search_energyplus_reference(query: str) -> str:
        """Search the EnergyPlus reference database for material properties,
        construction assemblies, schedule profiles, or design-day parameters.

        Call this tool BEFORE inventing property values when you need:
        - Thermal properties for a named material (conductivity, density,
          specific heat, R-value, roughness)
        - Standard construction layer sequences (outside → inside order)
        - Schedule type limit bounds or reference compact schedule profiles
        - Design-day meteorological parameters for HVAC sizing

        Args:
            query: Natural-language description of what you are looking for.
                   Be specific — include material name, building type, climate
                   zone, or parameter names. Examples:
                     'thermal conductivity of normal weight concrete 200 mm'
                     'standard exterior wall assembly brick insulation'
                     'medium office occupancy fraction schedule weekday'
                     'summer design day dry bulb temperature Beijing'

        Returns:
            JSON with a list of matching records, each containing:
            description, table_name, record_id, score (cosine similarity),
            and full_data with all EnergyPlus property values.
            Use the full_data fields as authoritative values instead of guessing.
            If no match is found (empty list or success=false), fall back to
            ASHRAE typical values and proceed.
        """
        if rag is None:
            return json.dumps(
                {
                    "success": False,
                    "message": (
                        "EnergyPlus reference database is unavailable. "
                        "Use ASHRAE default values and proceed."
                    ),
                    "data": None,
                }
            )

        try:
            results = []
            if allowed_tables:
                for table in allowed_tables:
                    hits = rag.search(
                        query=query,
                        top_k=top_k,
                        chunk_type=table,
                        score_threshold=score_threshold,
                    )
                    results.extend(hits)
                results.sort(key=lambda r: r.score or 0.0, reverse=True)
                results = results[:top_k]
            else:
                results = rag.search(
                    query=query,
                    top_k=top_k,
                    chunk_type=None,
                    score_threshold=score_threshold,
                )

            if not results:
                return json.dumps(
                    {
                        "success": True,
                        "message": (
                            "No matching records found in the EnergyPlus reference "
                            "database. Use ASHRAE default values and proceed."
                        ),
                        "data": [],
                    }
                )

            records = [
                {
                    "description": r.description,
                    "table_name": r.table_name,
                    "record_id": r.record_id,
                    "score": round(r.score or 0.0, 4),
                    "full_data": r.full_data,
                }
                for r in results
            ]
            return json.dumps(
                {
                    "success": True,
                    "message": f"Found {len(records)} matching EnergyPlus reference records.",
                    "data": records,
                }
            )

        except Exception as exc:
            return json.dumps(
                {
                    "success": False,
                    "message": (
                        f"RAG query failed: {exc}. "
                        "Use ASHRAE default values and proceed."
                    ),
                    "data": None,
                }
            )

    return search_energyplus_reference
