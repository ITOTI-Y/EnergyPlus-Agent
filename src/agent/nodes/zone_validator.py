"""Independent LLM-based completeness validator for the zone phase.

After the zone ReAct agent finishes, this module inspects whether the zones
actually created satisfy the ``zone_specs`` task. It runs as its own ReAct
subgraph (built via :func:`build_react_agent`) with two terminal tools —
``approve`` and ``reject`` — so the validator LLM must call one of them to
reach a verdict (``tools_condition`` keeps looping the LLM until it does).

The verdict flows back to :func:`zone_agent`, which on ``reject`` feeds the
reasons to the main zone LLM as a ``HumanMessage`` and re-invokes it, up to
``MAX_ZONE_VALIDATION_ROUNDS`` rounds.

Why a separate validator (rather than trusting the main agent's own
``list_zones`` self-check)? The main agent's LLM can silently produce zero
tool calls when the upstream LLM gateway misbehaves (empty / malformed
replies), leaving the pipeline with 0 zones but no error. This validator is
a second pair of eyes that explicitly compares specs vs. created zones and
fails loudly on any mismatch, catching that failure mode.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool

from src.agent.react import ReactState, build_react_agent
from src.mcp.state import ConfigState, _idf_values

# Verdict tuple shape returned by run_zone_validator:
#   ("approved", None)            -> zones satisfy the specs
#   ("rejected", [reason, ...])   -> reasons describe what's wrong
ValidatorVerdict = tuple[str, list[str] | None]

# Cap on ReAct iterations inside the validator subgraph. The validator has
# exactly two tools (approve/reject) and a prompt that forces calling one;
# this limit is a safety net against an LLM that refuses to call a tool.
# Each LLM<->tool roundtrip is 2 graph steps, so 25 allows ~12 rounds.
_VALIDATOR_RECURSION_LIMIT = 25

ZONE_VALIDATOR_SYSTEM_PROMPT = """You are a strict zone-completeness auditor for EnergyPlus.

You are given TWO inputs:
1. ZONE SPECS (the task) — the natural-language instructions describing every \
thermal zone that MUST be created (count, names, floor origin, multiplier, etc.).
2. ZONES ACTUALLY CREATED — the JSON list of Zone objects the builder agent \
produced (may be empty if it failed silently).

Audit the created zones against the specs on ALL FOUR criteria:
1. COUNT: Does the number of created zones match what the specs require? An \
   empty list when the specs ask for zones is a hard FAIL.
2. NAMES: Does every zone name the specs imply appear (case-exact) among the \
   created zones? Names drive every downstream cross-reference (surfaces, \
   HVAC, people, lights) — a missing or misspelled name breaks the whole model.
3. GEOMETRY: Are z_origin (floor elevation: ground floor = 0, upper floors = \
   floor-to-floor height) and multiplier (1 unless a typical floor is \
   explicitly duplicated) sensible for each zone?
4. COMPLETENESS: No duplicate zone names; no zone required by the specs but \
   missing; no obvious leftover/placeholder.

Decision protocol — you MUST call EXACTLY ONE tool, never neither:
- If ALL four criteria pass (or only trivial warnings remain), call \
  `approve(reason="<one-line why it's OK>")`.
- If ANY criterion fails, call `reject(reasons=["<specific, actionable \
  problem>", ...])`. Each reason must name the offending zone / spec so the \
  builder can fix it (e.g. "Specs require zone 'F1_Office_North' but it was \
  not created", "0 zones created but specs require 8", \
  "duplicate name 'Core' appears twice").

Do NOT output only text — you must call `approve` or `reject`. If unsure, \
re-read the ZONE SPECS and ZONES ACTUALLY CREATED sections, then decide.
"""


def _ok(msg: str, data=None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data=None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def make_zone_validator_tools() -> list[BaseTool]:
    """Build the two terminal verdict tools for the validator ReAct agent.

    Both follow the project-wide ``{"success": bool, "message", "data"}``
    JSON convention so the verdict can be parsed from the resulting
    ``ToolMessage`` identically to how ``detect_upstream_gap`` reads tool
    output. ``reject``'s data intentionally carries NO ``missing_ref`` key,
    so it is never mistaken for an upstream back-hop gap.
    """

    @tool
    def approve(reason: str) -> str:
        """Approve the created zones — they satisfy the zone_specs.

        Call this when all four audit criteria (count / names / geometry /
        completeness) pass. Provide a one-line reason summarizing why.

        Args:
            reason: One-line justification for approval.
        """
        return _ok("zones approved", {"reason": reason})

    @tool
    def reject(reasons: list[str]) -> str:
        """Reject the created zones — they do NOT satisfy the zone_specs.

        Call this when ANY audit criterion fails. Each reason must be
        specific and actionable, naming the offending zone / spec so the
        builder agent can fix it.

        Args:
            reasons: List of specific, actionable problems found.
        """
        return _err("zones rejected", {"reasons": reasons})

    return [approve, reject]


def _created_zones(local: ConfigState) -> list[dict[str, Any]]:
    """Serialize the zones currently in ``local`` for the validator prompt."""
    return [
        z.model_dump(exclude_none=True)
        for z in _idf_values(local.idf, "Zone")
    ]


def _parse_verdict(result: dict[str, Any]) -> ValidatorVerdict:
    """Extract the (decision, reasons) from the validator ReAct result.

    Scans messages for the last ``ToolMessage`` (the approve/reject call),
    parses its JSON, and maps success -> approved, failure -> rejected.
    Falls back to ("rejected", [generic]) if no verdict tool was called —
    this should not happen given the prompt forces a tool call, but guards
    against a misbehaving LLM so the caller never sees an ambiguous state.
    """
    last_tool_content: str | None = None
    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", None) == "tool":
            last_tool_content = msg.content
            break
    if last_tool_content is None:
        return ("rejected", ["Validator did not issue an approve/reject verdict."])
    try:
        payload = json.loads(last_tool_content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ("rejected", [f"Validator verdict unparseable: {last_tool_content!r}"])
    data = payload.get("data") or {}
    if payload.get("success"):
        return ("approved", None)
    reasons = data.get("reasons") if isinstance(data, dict) else None
    if not reasons:
        reasons = [payload.get("message", "Validator rejected without reasons.")]
    return ("rejected", list(reasons))


def run_zone_validator(
    zone_specs: str,
    local: ConfigState,
    llm: BaseChatModel,
) -> ValidatorVerdict:
    """Run the zone-completeness validator and return its verdict.

    Builds a fresh ReAct subgraph with the two verdict tools, feeds it the
    zone_specs (task) plus the zones actually created (read live from
    ``local``), and returns ``("approved", None)`` or
    ``("rejected", [reasons])``.

    Args:
        zone_specs: The original zone-creation task (from intake_output).
        local: The phase-local ConfigState the main zone agent mutated; the
            validator reads (never writes) the zones from it.
        llm: A BaseChatModel instance (reused from the main zone agent to
            avoid re-parsing the LLM config YAML).
    """
    zones_created = _created_zones(local)
    content = (
        "ZONE SPECS (the task the builder was given):\n"
        f"{zone_specs}\n\n"
        "ZONES ACTUALLY CREATED (read from the model):\n"
        f"{json.dumps(zones_created, indent=2, ensure_ascii=False, default=str)}"
    )
    agent = build_react_agent(
        llm=llm,
        tools=make_zone_validator_tools(),
        system_prompt=ZONE_VALIDATOR_SYSTEM_PROMPT,
    )
    result = agent.invoke(
        ReactState(messages=[HumanMessage(content=content)]),
        config={"recursion_limit": _VALIDATOR_RECURSION_LIMIT},
    )
    return _parse_verdict(result)
