"""Shared helpers for phase-agent nodes.

Kept here (rather than `src/agent/_share.py`) because the scope is
nodes-internal — no other part of the agent package uses these.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, Field

from src.agent._share import language_directive
from src.agent.phases import PHASE_TYPES, Phase, owner
from src.modeling.validation import ModelIssue, model_issues
from src.state.config_state import ConfigState

if TYPE_CHECKING:
    from src.agent.state import AgentState

MAX_SELF_REPAIR_ROUNDS: Final = 2
"""Max extra invokes per phase for cross-ref self-repair.

Two rounds is enough for the LLM to see its own error feedback and
react; repeated failures beyond that point usually mean the intake
specs are broken, which the outer validate loop handles better.
"""


class MissingInput(BaseModel):
    """An upstream object a phase needed but did not find."""

    kind: Literal["zone", "material", "construction", "schedule", "surface"] = Field(
        description="Kind of object; the phase of that name creates it"
    )
    description: str = Field(
        description="What is needed and for what, e.g. 'a window construction "
        "(kind window) for the office windows'"
    )


class PhaseReport(BaseModel):
    """Final answer of a phase that references upstream objects."""

    missing_inputs: list[MissingInput] = Field(
        default_factory=list,
        description="Needed objects that do not exist; empty when none",
    )


def missing_input_issues(phase: Phase, report: PhaseReport | None) -> list[ModelIssue]:
    """Reported missing inputs, each blamed on the phase that creates it.

    Validation then reruns that phase with the request as feedback, and the
    reporting phase after it.
    """
    if report is None:
        return []
    return [
        ModelIssue(
            PHASE_TYPES[missing.kind][0].idf_object_type(),
            None,
            None,
            f"The {phase} phase needs {missing.description}, which does not "
            "exist; create it.",
        )
        for missing in report.missing_inputs
    ]


def skipped(state: AgentState, phase: Phase) -> bool:
    """Whether the phase sits out this pass; its objects stay as they are."""
    return phase not in state.pending_phases


def with_feedback(specs: str, state: AgentState, phase: Phase) -> str:
    """The phase task, plus the problems its objects had in the previous pass."""
    problems = state.phase_feedback.get(phase)
    if not problems:
        return specs
    listed = "\n".join(f"  - {p}" for p in problems)
    return (
        f"{specs}\n\nThe objects this phase built in the previous attempt had "
        f"these problems, and were removed. Build them again without them:\n"
        f"{listed}"
    )


def last_message_text(result: dict[str, Any]) -> str:
    """Final AI message of an agent run, used when it gave no structured answer."""
    message = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
    )
    return message.text if message is not None else "no response"


def invoke_with_self_repair(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    local_config: ConfigState,
    specs: str,
    *,
    phase: Phase,
) -> dict[str, Any]:
    """Run a phase agent and make it repair problems in its own objects.

    After each `agent.invoke`, check the model in code (the LLM cannot skip
    it) and send back the problems blamed on object types this phase owns.
    Loop up to MAX_SELF_REPAIR_ROUNDS.

    Problems in other phases' objects are left to the cross-reference nodes:
    this phase has no tools to fix them. A problem in its own object may
    still name a missing upstream object, which the LLM should report rather
    than fabricate.

    Args:
        agent: Compiled agent graph from `build_agent`.
        local_config: The deep-copied ConfigState the phase mutates.
        specs: Natural-language task for the phase (from intake_output).
        phase: Phase whose objects are checked; also used in logs.

    Returns:
        The final agent result dict (shape {"messages": [...]}, plus
        "structured_response" when the agent declares a response_format).
    """
    messages: list[AnyMessage] = [HumanMessage(content=specs)]

    for attempt in range(MAX_SELF_REPAIR_ROUNDS + 1):
        result = agent.invoke({"messages": messages})
        errors = [i for i in model_issues(local_config.idf) if owner(i) == phase]

        if not errors:
            if attempt > 0:
                logger.info("[{}] self-repair succeeded on round {}", phase, attempt)
            return result

        if attempt == MAX_SELF_REPAIR_ROUNDS:
            logger.warning(
                "[{}] self-repair exhausted after {} rounds, {} errors remain "
                "— escalating to outer validate loop",
                phase,
                MAX_SELF_REPAIR_ROUNDS,
                len(errors),
            )
            return result

        logger.info(
            "[{}] self-repair round {}: {} cross-ref errors",
            phase,
            attempt + 1,
            len(errors),
        )
        feedback = HumanMessage(
            content=(
                "Validation found problems in the objects you created:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nFix the objects YOU just created: `delete_<x>` the "
                "broken object, then `create_<x>` it again. If the broken "
                "reference names an upstream "
                "resource (zone / schedule / material / construction / "
                "surface) that truly does not exist, list it in the "
                "`missing_inputs` of your final answer and do NOT fabricate "
                "a replacement — "
                "upstream phases own those objects." + language_directive()
            )
        )
        messages = [*list(result["messages"]), feedback]

    return result


MAX_STRUCTURED_ATTEMPTS: Final = 2
"""A model replying with text instead of the tool call usually complies
when told so once."""


def without_empty_choices(node: Any) -> Any:
    """The schema with "" removed from enums.

    idfpy allows "" (leave blank) in many choices; Gemini rejects empty enum
    values in tool declarations. Replies are still validated by the model.
    """
    if isinstance(node, dict):
        return {
            key: [v for v in value if v != ""]
            if key == "enum" and isinstance(value, list)
            else without_empty_choices(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [without_empty_choices(v) for v in node]
    return node


def strict_tool(schema: type[BaseModel]) -> dict[str, Any]:
    """The schema as a strict tool: every field required, no extra keys.

    Without strict mode Haiku now and then left out required specs (e.g.
    construction_specs) and repeated the omission when told.
    """
    return without_empty_choices(convert_to_openai_tool(schema, strict=True))


_LEAKED_FIELD_START = re.compile(r'<(?:parameter name="(\w+)"|(\w+))>')


def unpack_leaked_fields(args: dict[str, Any], fields: Iterable[str]) -> list[str]:
    """Move fields written as XML into a string field back to their own keys.

    Haiku sometimes ends a string argument with its closing tag and writes
    the following arguments in its native tool format inside it, e.g.
    ``'...</lights_specs>\n<parameter name="equipment_specs">...'``, leaving
    those fields missing. The text after the closing tag is cut off; each
    field opened there runs to the next field's opening tag or the end, less
    its closing tag, and fills its key unless the key is set (the first of
    repeated fields wins).

    Returns:
        The fields filled, in ``args``, which is changed in place.
    """
    known = set(fields)
    filled: list[str] = []
    for key in list(args):
        value = args[key]
        if not isinstance(value, str) or f"</{key}>" not in value:
            continue
        own, _, rest = value.partition(f"</{key}>")
        args[key] = own.strip()
        starts = [
            (m, name)
            for m in _LEAKED_FIELD_START.finditer(rest)
            if (name := m[1] or m[2]) in known
        ]
        for i, (match, name) in enumerate(starts):
            stop = starts[i + 1][0].start() if i + 1 < len(starts) else len(rest)
            text = rest[match.end() : stop].partition(f"</{name}>")[0]
            if name not in args:
                args[name] = text.strip()
                filled.append(name)
    return filled


def structured[T: BaseModel, R](
    llm: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    check: Callable[[T], R],
) -> tuple[T, R]:
    """Call ``llm`` for ``schema`` as a forced tool call, retrying once.

    The tool is the schema without empty enum values (see
    ``without_empty_choices``); the arguments are validated with the full
    model. ``check`` turns the parsed reply into the result the caller
    needs; a ValueError from it, like invalid arguments, is sent back to the
    LLM for one more attempt.

    Raises:
        RuntimeError: If no attempt gives a usable reply.
    """
    caller = llm.with_structured_output(
        strict_tool(schema), method="function_calling", include_raw=True, strict=True
    )
    problem = ""
    for _ in range(MAX_STRUCTURED_ATTEMPTS):
        result = cast(dict[str, Any], caller.invoke(messages))
        try:
            if result.get("parsed") is None:
                raw: BaseMessage | None = result.get("raw")
                raise ValueError(
                    f"no {schema.__name__} tool call (parsing error "
                    f"{result.get('parsing_error')!r}; reply "
                    f"{repr(raw.content if raw is not None else raw)[:300]})"
                )
            args = dict(result["parsed"])
            if filled := unpack_leaked_fields(args, schema.model_fields):
                logger.warning(
                    "{}: fields written inside others: {}", schema.__name__, filled
                )
            parsed = schema.model_validate(args)
            return parsed, check(parsed)
        except ValueError as e:  # pydantic.ValidationError is a ValueError
            problem = str(e)
        logger.warning("{}: unusable reply: {}", schema.__name__, problem)
        messages = [
            *messages,
            HumanMessage(
                content=f"Your reply was unusable: {problem}. Call the "
                f"{schema.__name__} tool again with a corrected argument."
            ),
        ]
    raise RuntimeError(f"no usable {schema.__name__}: {problem}")
