"""Shared helpers for phase-agent nodes.

Kept here (rather than `src/agent/_share.py`) because the scope is
nodes-internal — no other part of the agent package uses these.
"""

from __future__ import annotations

from typing import Any, Final

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from src.agent._share import language_directive
from src.agent.react import ReactState
from src.mcp.state import ConfigState

MAX_SELF_REPAIR_ROUNDS: Final = 2
"""Max extra invokes per phase for cross-ref self-repair.

Two rounds is enough for the LLM to see its own error feedback and
react; repeated failures beyond that point usually mean the intake
specs are broken, which the outer validate loop handles better.
"""

REVISION_PREFIX: Final[str] = (
    "IMPORTANT — REVISION MODE. You are MODIFYING an existing model that was "
    "built in a previous turn. The model already contains objects.\n\n"
    "Before doing anything, call the `list_*` tool to see what already "
    "exists (by exact name). Then:\n"
    "- To change an existing object: use `update_*` (PREFERRED — never "
    "delete+recreate an object that only needs a field change).\n"
    "- To remove an object: use `delete_*`.\n"
    "- To add a brand-new object: use `create_*`.\n"
    "- DO NOT recreate objects that already exist and need no change.\n"
    "- If the spec says 'no changes needed', call `list_*` once to confirm "
    "then return immediately without creating/updating/deleting anything.\n\n"
    "Apply only the modifications described below:\n\n"
)
"""Prefix prepended to phase specs when ``is_revision`` is True, steering
phase agents toward incremental update/delete over full recreation."""

VALIDATION_ERROR_HEADER: Final[str] = (
    "=== GLOBAL VALIDATION ERRORS (from validate node) ===\n"
    "These cross-reference errors survived the previous pipeline run. "
    "You are being re-invoked as a DIRECTED ROLLBACK to fix the objects "
    "YOU own. Use `update_*` / `delete_*` to repair the broken references "
    "below — do NOT recreate objects that are otherwise correct, and do "
    "NOT touch objects owned by other phases.\n"
)

# Pipeline order, earliest first. When errors span multiple phases we
# roll back to the earliest one: fixing upstream often resolves downstream
# refs as a side effect (and downstream phases re-run anyway via normal
# graph edges from the rollback target).
PIPELINE_ORDER: Final[tuple[str, ...]] = (
    "zone",
    "material",
    "schedule",
    "construction",
    "surface",
    "fenestration",
    "hvac",
    "people",
    "lights",
)

# Maps a substring of an error message to the phase that OWNS the broken
# reference. validate_references() phrases errors as
# "<Owner> '<name>' references <kind> '<ref>' which does not exist", so
# the owner is the safest single-hop rollback target: that phase has its
# own objects in memory and can rename / delete / rebuild them via the
# update_* tools.
_ERROR_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("Ideal load system", "hvac"),
    ("Thermostat '", "hvac"),
    ("Construction '", "construction"),
    ("Surface '", "surface"),
    ("Fenestration '", "fenestration"),
    ("People '", "people"),
    ("Lights '", "lights"),
)


def apply_revision_prefix(specs: str, is_revision: bool) -> str:
    """Prepend the revision-mode instruction prefix when applicable."""
    return REVISION_PREFIX + specs if is_revision else specs


def classify_errors(errors: list[str]) -> dict[str, list[str]]:
    """Group validation errors by the phase that owns the broken reference.

    Returns ``{phase: [errors]}``. Errors that match no known pattern are
    dropped — the caller falls back to a full re-intake for those rather
    than risking a wrong directed hop.
    """
    grouped: dict[str, list[str]] = {}
    for err in errors:
        for needle, phase in _ERROR_PATTERNS:
            if needle in err:
                grouped.setdefault(phase, []).append(err)
                break
    return grouped


def earliest_phase(phases: set[str]) -> str | None:
    """Return the earliest phase in PIPELINE_ORDER present in *phases*."""
    for phase in PIPELINE_ORDER:
        if phase in phases:
            return phase
    return None


def inject_validation_errors(specs: str, errors: list[str]) -> str:
    """Append validation-error context to phase specs.

    Used during directed rollback: validate_node routes back to a phase
    and surfaces the global errors so the phase's agent knows exactly
    which references to repair via its update_* tools. No-op when there
    are no errors (normal first-run / clean-revision path).
    """
    if not errors:
        return specs
    bullet = "\n".join(f"  - {e}" for e in errors)
    return f"{specs}\n\n{VALIDATION_ERROR_HEADER}{bullet}\n"


def _errors_owned_by_phase(errors: list[str], phase: str) -> list[str]:
    """Return only the errors owned by *phase* (per classify_errors).

    Used by self-repair during directed rollback so the convergence
    check only looks at THIS phase's errors. Errors owned by other
    phases are out of scope and would otherwise keep the loop alive
    forever.
    """
    grouped = classify_errors(errors)
    return grouped.get(phase, [])


def invoke_with_self_repair(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    local_config: ConfigState,
    specs: str,
    *,
    phase: str,
    is_revision: bool = False,
    validation_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Run a phase ReAct agent and force cross-reference self-repair.

    After each `agent.invoke`, call `local_config.validate_references()`
    in code (not tool — cannot be skipped by the LLM). If errors exist,
    push them back as a HumanMessage and invoke again. Loop up to
    MAX_SELF_REPAIR_ROUNDS.

    Two scoping modes:
    - Normal first-run / clean-revision path (validation_errors is None):
      any cross-ref error triggers a self-repair round, since the phase
      just created its objects and any broken reference is its own.
    - Directed rollback (validation_errors supplied by the validate node):
      local_config already contains objects from every phase, so some
      errors are owned by *other* phases and out of scope. The convergence
      check then only considers errors owned by THIS phase, and the
      feedback message lists only this phase's errors.

    Args:
        agent: Compiled ReAct subgraph from `build_react_agent`.
        local_config: The deep-copied ConfigState the phase mutates.
        specs: Natural-language task for the phase (from intake_output).
        phase: Name used in logs ("construction", "surface", ...).
        is_revision: When True, prepend REVISION_PREFIX to steer the agent
            toward update/delete over full recreation.
        validation_errors: Optional global errors injected by the validate
            node during a directed rollback. Prepended to specs so the
            phase knows exactly which of its own references to repair,
            AND scopes the self-repair convergence check to this phase.

    Returns:
        The final ReAct result dict (shape {"messages": [...]}).
    """
    full_specs = apply_revision_prefix(specs, is_revision)
    if validation_errors:
        full_specs = inject_validation_errors(full_specs, validation_errors)
    messages: list[AnyMessage] = [HumanMessage(content=full_specs)]

    is_rollback = bool(validation_errors)

    def _scoped_errors(errs: list[str]) -> list[str]:
        """Errors this phase is responsible for under the current mode."""
        return _errors_owned_by_phase(errs, phase) if is_rollback else errs

    for attempt in range(MAX_SELF_REPAIR_ROUNDS + 1):
        result = agent.invoke(ReactState(messages=messages))
        all_errors = local_config.validate_references()
        scoped = _scoped_errors(all_errors)

        if not scoped:
            if attempt > 0:
                logger.info(
                    "[{}] self-repair succeeded on round {}", phase, attempt
                )
            return result

        if attempt == MAX_SELF_REPAIR_ROUNDS:
            logger.warning(
                "[{}] self-repair exhausted after {} rounds, {} in-scope "
                "errors remain — escalating to outer validate loop",
                phase,
                MAX_SELF_REPAIR_ROUNDS,
                len(scoped),
            )
            return result

        logger.info(
            "[{}] self-repair round {}: {} in-scope cross-ref errors",
            phase,
            attempt + 1,
            len(scoped),
        )
        feedback = HumanMessage(
            content=(
                "Cross-reference validation failed for objects YOU own:\n"
                + "\n".join(f"  - {e}" for e in scoped)
                + "\n\nFix them: use `update_<x>` to rename references, or "
                "`delete_<x>` + `create_<x>` to rebuild. If the broken "
                "reference names an upstream resource (zone / schedule / "
                "material / construction / surface) that truly does not "
                "exist, report it in your final message and do NOT "
                "fabricate a replacement — upstream phases own those "
                "objects." + language_directive()
            )
        )
        messages = [*list(result["messages"]), feedback]

    return result
