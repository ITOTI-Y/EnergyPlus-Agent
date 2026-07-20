"""Shared helpers for phase-agent nodes.

Kept here (rather than `src/agent/_share.py`) because the scope is
nodes-internal — no other part of the agent package uses these.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from src.agent._share import language_directive
from src.agent.react import ReactState
from src.agent.state import AgentState
from src.mcp.state import ConfigState, _idf_values


def clone_for_phase(state: AgentState) -> ConfigState:
    """Clone the state's config_state for in-place phase mutation, ensuring
    the seed model is present.

    On revision turns, LangGraph's input coercion strips ConfigState's
    PrivateAttr ``_idf`` at the graph START and at each checkpoint write.
    The seed model survives as the declared ``seed_idf_text`` field, and
    ``recover_idf_from_seed()`` rebuilds ``_idf`` from it. We clone AFTER
    recovery so the clone carries the recovered IDF, and we also keep
    seed_idf_text on the clone so subsequent recoveries work if the
    clone's IDF is stripped again downstream.
    """
    state.config_state.recover_idf_from_seed()
    return state.config_state.clone()


MAX_SELF_REPAIR_ROUNDS: Final = 2
"""Max extra invokes per phase for cross-ref self-repair.

Two rounds is enough for the LLM to see its own error feedback and
react; repeated failures beyond that point usually mean the intake
specs are broken, which the outer validate loop handles better.
"""

HOP_LIMIT: Final[int] = 3
"""Max back-hops per session. Phase nodes refuse to issue another
``Command(goto=<earlier phase>)`` once ``state.hop_count`` reaches this,
falling through to the outer validate loop instead. Prevents infinite
A->B->A ping-pong between phases."""

# Maps a tool-reported ``missing_ref`` object type to the phase that
# OWNS (should have created) that object. Used by :func:`detect_upstream_gap`
# to decide where to back-hop when a downstream phase finds a missing
# upstream dependency at tool-call time.
_MISSING_REF_TO_PHASE: Final[dict[str, str]] = {
    "Construction": "construction",
    "Material": "material",
    "WindowMaterial:SimpleGlazingSystem": "material",
    "BuildingSurface:Detailed": "surface",
    "Zone": "zone",
    "Schedule:Compact": "schedule",
}

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
PIPELINE_ORDER: Final[tuple[PIPELINE_PHASE, ...]] = (
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

PIPELINE_PHASE: Final = Literal[
    "zone",
    "material",
    "schedule",
    "construction",
    "surface",
    "fenestration",
    "hvac",
    "people",
    "lights",
]

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


def _is_upstream(target: str, current: str) -> bool:
    """True if *target* phase runs before *current* in PIPELINE_ORDER.

    Back-hops are only legal toward an earlier phase; a phase must never
    hop to itself or to a later phase.
    """
    try:
        return PIPELINE_ORDER.index(target) < PIPELINE_ORDER.index(current)
    except ValueError:
        return False


# Maps a substring of ``validate_references()`` error text (the
# ``references <kind> '<name>'`` part) to ``(ref_type, owning_phase)``.
# This is the IDF-grounded counterpart of ``_MISSING_REF_TO_PHASE``: instead
# of trusting tool-reported ``missing_ref`` payloads (which stay forever in
# the message history and cause stale-gap false back-hops), we re-derive the
# gap from the LIVE cross-ref check on the IDF itself. A reference described
# as "references zone 'X'" that "does not exist" means phase 'zone' owes us
# 'X'; "references construction 'Y'" means 'construction' owes us 'Y'; etc.
_REF_KIND_TO_PHASE: Final[tuple[tuple[str, str, str], ...]] = (
    # Order matters only for specificity (longer/unique needles first).
    ("references heating setpoint schedule", "Schedule:Compact", "schedule"),
    ("references cooling setpoint schedule", "Schedule:Compact", "schedule"),
    ("references availability schedule", "Schedule:Compact", "schedule"),
    ("references schedule", "Schedule:Compact", "schedule"),
    ("references building surface", "BuildingSurface:Detailed", "surface"),
    ("references construction", "Construction", "construction"),
    ("references material", "Material", "material"),
    ("references zone", "Zone", "zone"),
    # Thermostat refs are owned by hvac itself (same phase), so they are NOT
    # a back-hop target and intentionally omitted — they self-repair locally.
)


def detect_upstream_gap(result: dict[str, Any], phase: str) -> dict[str, str] | None:
    """Scan a ReAct result for a 'missing upstream object' tool error.

    .. deprecated-paths::
        This reads tool-reported ``missing_ref`` payloads from the message
        history. Because ReAct's ``add_messages`` reducer accumulates every
        prior round's ToolMessages, a gap reported on round 0 is STILL
        visible on round N even after the LLM successfully self-healed —
        producing a false back-hop. ``invoke_with_self_repair`` therefore
        uses :func:`detect_upstream_gap_from_state` (IDF-grounded) instead.
        This function is retained for any caller that wants the raw
        tool-signal view.

    Tools report reference failures as ``{"success": false, "data":
    {"missing_ref": <object type>, "missing_name": <name>}}``. When such
    an error points at an object owned by an **earlier** phase, we can
    back-hop to that phase and have it create the missing object,
    instead of aborting the pipeline.

    Args:
        result: The dict returned by ``agent.invoke(ReactState(...))`` —
            shape ``{"messages": [...]}``.
        phase: Name of the current phase (e.g. "fenestration").

    Returns:
        ``{"target": <phase>, "missing_ref": <type>, "missing_name": <name>}``
        for the first upstream gap found, or ``None``.
    """
    for msg in result.get("messages", []):
        # ToolMessage has .type == "tool"; guard against other message kinds.
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            payload = json.loads(msg.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("success"):
            continue
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            continue
        ref_type = data.get("missing_ref")
        if not ref_type:
            continue
        target = _MISSING_REF_TO_PHASE.get(str(ref_type))
        if not target:
            continue
        if _is_upstream(target, phase):
            return {
                "target": target,
                "missing_ref": str(ref_type),
                "missing_name": str(data.get("missing_name", "")),
            }
    return None


def detect_upstream_gap_from_state(
    local_config: ConfigState,
    phase: str,
    errors: list[str] | None = None,
) -> dict[str, str] | None:
    """Detect a missing-upstream-object gap from the LIVE IDF.

    Unlike :func:`detect_upstream_gap` (which scans the accumulated message
    history and can report a stale gap that the LLM already fixed), this
    re-runs ``validate_references()`` against the current model and asks the
    IDF itself whether the missing object still exists. A gap is reported
    only when:

    1. an object created by THIS phase references a name that does not
       exist, AND
    2. the referenced name's owning phase runs strictly earlier in the
       pipeline (so a back-hop is legal), AND
    3. the referenced name is genuinely absent from the model right now
       (guards against a wrong-name self-heal that picked an existing-but-
       unrelated object — though that case is rare; the existence check is
       the primary guarantee).

    Args:
        local_config: The phase-local ConfigState the agent is mutating.
            Only used to recompute errors when *errors* is None.
        phase: Name of the current phase.
        errors: Optionally, an already-computed ``validate_references()``
            result. Pass this when the caller has just run the same check
            to avoid scanning the IDF twice per repair round.

    Returns:
        ``{"target", "missing_ref", "missing_name"}`` for the first live
        upstream gap, or ``None``.
    """
    if errors is None:
        errors = local_config.validate_references()
    if not errors:
        return None
    # Only errors whose OWNER is this phase can represent an upstream gap
    # for it — a dangling ref on someone else's object is their problem.
    owned = _errors_owned_by_phase(errors, phase)
    if not owned:
        return None
    for err in owned:
        for needle, ref_type, target in _REF_KIND_TO_PHASE:
            idx = err.find(needle)
            if idx < 0:
                continue
            tail = err[idx + len(needle) :]
            # tail looks like " 'F1_Office' which does not exist." — pull the
            # first single-quoted token, which is the missing reference name.
            name = _first_quoted(tail)
            if not name:
                continue
            if not _is_upstream(target, phase):
                continue
            return {
                "target": target,
                "missing_ref": ref_type,
                "missing_name": name,
            }
    return None


def _first_quoted(text: str) -> str | None:
    """Return the contents of the first single-quoted span in *text*."""
    start = text.find("'")
    if start < 0:
        return None
    end = text.find("'", start + 1)
    if end < 0:
        return None
    return text[start + 1 : end]


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

        # A missing upstream object is first fed back to the phase LLM so it
        # can try to self-heal by using existing names from list_* tools. Only
        # after the repair budget is exhausted do we surface hop_request to the
        # caller and let the existing maybe_backhop() mechanism route upstream.
        #
        # NOTE: the gap is derived from the LIVE IDF (validate_references),
        # NOT from the message history. A tool-reported missing_ref payload
        # stays in add_messages' accumulated history forever, so scanning
        # history would re-report a gap the LLM already fixed on an earlier
        # round and produce a false back-hop. Asking the IDF is authoritative.
        all_errors = local_config.validate_references()
        scoped = _scoped_errors(all_errors)
        # Reuse the already-computed errors for gap detection instead of
        # re-running validate_references() (it scans the whole IDF).
        gap = detect_upstream_gap_from_state(local_config, phase, all_errors)

        # P4 surface 0-output guard: if the surface phase has produced ZERO
        # BuildingSurface:Detailed objects and still has repair budget, force
        # another self-repair round with a pointed "you've built nothing yet"
        # nudge. This stops surface from back-hopping on the first missing
        # zone/construction when it could simply build every surface whose
        # upstream already exists. Without this, a transient missing ref can
        # leave the model with 0 surfaces entering simulation.
        surface_empty = phase == "surface" and not _idf_values(
            local_config.idf, "BuildingSurface:Detailed"
        )

        if not scoped and not gap and not surface_empty:
            if attempt > 0:
                logger.info("[{}] self-repair succeeded on round {}", phase, attempt)
            return result

        if attempt == MAX_SELF_REPAIR_ROUNDS:
            if gap:
                result["hop_request"] = gap
                logger.info(
                    "[{}] upstream gap persisted after {} repair rounds "
                    "(missing {} '{}') -> hop to {}",
                    phase,
                    MAX_SELF_REPAIR_ROUNDS,
                    gap["missing_ref"],
                    gap["missing_name"],
                    gap["target"],
                )
                return result
            logger.warning(
                "[{}] self-repair exhausted after {} rounds, {} in-scope "
                "errors remain — escalating to outer validate loop",
                phase,
                MAX_SELF_REPAIR_ROUNDS,
                len(scoped),
            )
            return result

        logger.info(
            "[{}] self-repair round {}: {} in-scope cross-ref errors{}{}",
            phase,
            attempt + 1,
            len(scoped),
            " plus upstream gap" if gap else "",
            " plus 0 surfaces built" if surface_empty else "",
        )
        feedback = HumanMessage(
            content=_build_repair_feedback(
                scoped=scoped,
                gap=gap,
                surface_empty=surface_empty,
            )
        )
        messages = [*list(result["messages"]), feedback]

    return result


def _build_repair_feedback(
    *,
    scoped: list[str],
    gap: dict[str, str] | None,
    surface_empty: bool,
) -> str:
    """Compose the self-repair HumanMessage body.

    Three orthogonal failure signals can drive a repair round, and the
    preamble must describe the one(s) that actually fired so the LLM is not
    told "cross-reference validation failed" when in fact it only produced
    zero output (the P4 surface case). The closing instructions stay
    generic enough to apply in all three cases.
    """
    parts: list[str] = []

    if scoped:
        parts.append("Cross-reference validation failed for objects YOU own:")
        parts.extend(f"  - {e}" for e in scoped)
    elif gap:
        # No stored-object reference errors remain, but a tool call failed
        # due to a missing upstream reference.
        parts.append(
            "No stored-object reference errors remain, but a tool call "
            "failed due to a missing upstream reference."
        )
    elif surface_empty:
        # P4 pure 0-output case: nothing is broken, the phase just hasn't
        # produced anything yet. Do NOT claim a cross-ref or upstream
        # failure — that would mislead the LLM into 'fixing' non-existent
        # errors instead of building.
        parts.append(
            "No errors were found, but this phase has produced zero output "
            "so far. You must create objects now."
        )

    if gap:
        parts.append("")
        parts.append("A reference points at a missing upstream object:")
        parts.append(
            f"  - missing {gap['missing_ref']} '{gap['missing_name']}' "
            f"(owned by the {gap['target']} phase)"
        )
        parts.append(
            "Before requesting upstream repair, try to recover locally: "
            "call the relevant list_* tools, use an existing exact name if "
            "one satisfies the spec, or update/delete any object you "
            "created with the bad reference. If the object truly does not "
            "exist after checking, report that clearly."
        )

    if surface_empty:
        parts.append("")
        parts.append(
            "You have not created ANY BuildingSurface:Detailed objects "
            "yet. Call list_zones and list_constructions to map the "
            "available names, then create ALL surfaces whose zone and "
            "construction already exist. Only request back-hop if a "
            "required upstream object is truly absent AFTER you have "
            "built every surface you can."
        )

    parts.append("")
    parts.append(
        "Fix them: use `update_<x>` to rename references, or "
        "`delete_<x>` + `create_<x>` to rebuild. If the broken reference "
        "names an upstream resource (zone / schedule / material / "
        "construction / surface) that truly does not exist, report it in "
        "your final message and do NOT fabricate a replacement — upstream "
        "phases own those objects." + language_directive()
    )
    return "\n".join(parts)


def build_upstream_specs(gap: dict[str, str], state: AgentState) -> str:
    """Build a natural-language instruction for the back-hop target phase.

    Combines: (a) the intake specs the target phase would normally work
    from (so it has context for the object it must create), and (b) a
    concrete "please create <missing_name>" instruction derived from the
    gap detected by the downstream phase.

    Args:
        gap: ``{"target", "missing_ref", "missing_name"}`` from
            :func:`detect_upstream_gap`.
        state: Current AgentState — used to pull the intake specs for the
            target phase.

    Returns:
        A specs string to inject into the target phase's task.
    """
    target = gap["target"]
    missing_name = gap["missing_name"]
    missing_ref = gap["missing_ref"]

    # Pull the intake specs the target phase would normally consume, so it
    # has full context (not just the missing-name hint).
    intake_specs = ""
    if state.intake_output is not None:
        field_map = {
            "material": "material_specs",
            "schedule": "schedule_specs",
            "construction": "construction_specs",
            "surface": "surface_specs",
            "zone": "zone_specs",
        }
        attr = field_map.get(target)
        if attr:
            intake_specs = getattr(state.intake_output, attr, "") or ""

    parts = [
        "=== UPSTREAM REQUEST (a downstream phase needs an object you "
        "should have created) ===",
        f"A downstream phase tried to reference {missing_ref} "
        f"'{missing_name}' but it does not exist yet. "
        f"Please CREATE it now (use create_* / do NOT recreate objects "
        f"that already exist). After creating it, return as usual.",
    ]
    if intake_specs:
        parts.append(
            f"\nFor reference, your original intake specs were:\n{intake_specs}"
        )
    return "\n".join(parts)


def maybe_backhop(
    result: dict[str, Any],
    state: AgentState,
    local: ConfigState,
    phase: str,
) -> Command | None:
    """If the ReAct result carries a back-hop request, build the Command.

    Called by each phase-2/3 node after ``invoke_with_self_repair``. If a
    back-hop is warranted (hop_request present and hop_count under the
    limit), returns a ``Command(goto=<earlier phase>)`` carrying the
    current config_state (so the upstream phase sees objects built so far),
    the upstream_request specs, and an incremented hop_count. Otherwise
    returns None and the caller proceeds with its normal return.

    The forward static edges from this phase are NOT taken when a Command
    is returned — LangGraph routes to the goto target and ignores them,
    which is exactly what we want for a back-hop. After the upstream phase
    finishes, normal graph edges carry flow forward again through the
    pipeline (construction -> surface -> fenestration -> ...).
    """
    gap = result.get("hop_request")
    if not gap:
        return None
    if state.hop_count >= HOP_LIMIT:
        logger.warning(
            "[{}] back-hop to {} suppressed: hop_count={} reached HOP_LIMIT={}",
            phase,
            gap["target"],
            state.hop_count,
            HOP_LIMIT,
        )
        return None

    specs_for_upstream = build_upstream_specs(gap, state)
    logger.info(
        "[{}] issuing back-hop Command -> {} (hop_count {}->{})",
        phase,
        gap["target"],
        state.hop_count,
        state.hop_count + 1,
    )
    return Command(
        goto=gap["target"],
        update={
            "config_state": local,
            "upstream_request": {
                "target": gap["target"],
                "specs": specs_for_upstream,
            },
            "hop_count": state.hop_count + 1,
            "is_revision": True,
        },
    )
