import json
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    ModelRequest,
    ModelResponse,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.agent._share import language_directive
from src.configs.config import LLMConfig

load_dotenv()


def load_llm_config() -> LLMConfig:
    raw = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "configs" / "llm.yaml"
    )
    return LLMConfig.model_validate(OmegaConf.to_container(raw, resolve=True))


def create_llm(config: LLMConfig | None = None) -> BaseChatModel:
    """Create a LangChain chat model from LLMConfig.

    Args:
        config: Optional override. If None, reads src/configs/llm.yaml.

    Returns:
        A BaseChatModel routed to the configured provider.
    """
    if config is None:
        config = load_llm_config()

    kwargs: dict[str, Any] = {
        "max_tokens": config.max_tokens,
        "max_retries": config.max_retries,
        "timeout": config.timeout,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.reasoning_max_tokens is not None:
        kwargs["extra_body"] = {
            "reasoning": {"max_tokens": config.reasoning_max_tokens}
        }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return init_chat_model(config.model_name, model_provider=config.provider, **kwargs)


def create_vision_llm(config: LLMConfig | None = None) -> BaseChatModel:
    """The chat model that reads input images: the configured one, renamed.

    It sends no temperature: Claude 5.5 models reject any, and reading a
    photo needs no sampling setting of its own.

    Raises:
        ValueError: If no vision model is configured (LLM_VISION_MODEL).
    """
    if config is None:
        config = load_llm_config()
    if config.vision_model_name is None:
        raise ValueError(
            "images need a vision model; set LLM_VISION_MODEL, or give the "
            "building in text only"
        )
    return create_llm(
        config.model_copy(
            update={"model_name": config.vision_model_name, "temperature": None}
        )
    )


@wrap_model_call
def _sequential_tool_calls(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Disable parallel tool calls so each call is validated sequentially.

    Phase tools mutate a shared (local-copy) ConfigState, so concurrent
    calls would race. Guarded on `request.tools` because the provider
    rejects `parallel_tool_calls` on a request that declares no tools.
    """
    if request.tools:
        request = request.override(
            model_settings={**request.model_settings, "parallel_tool_calls": False}
        )
    return handler(request)


WRITE_TOOL_PREFIXES: Final = ("create_", "update_", "delete_")


def serial_write_tools_middleware() -> AgentMiddleware:
    """Serialize write-tool execution within one agent instance.

    `_sequential_tool_calls` asks the provider for at most one tool call
    per round, but `parallel_tool_calls` is a hint that OpenAI-compatible
    endpoints may ignore. When a round still returns several tool calls,
    the ToolNode executes them concurrently, so create_/update_/delete_
    tools could race on the shared local ConfigState (e.g. the
    check-then-add in every create tool). Read tools pass through.
    """
    lock = threading.Lock()

    def _serialize(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = str(request.tool_call.get("name", ""))
        if name.startswith(WRITE_TOOL_PREFIXES):
            with lock:
                return handler(request)
        return handler(request)

    return wrap_tool_call(name="SerialWriteTools")(_serialize)


MAX_REPEATED_FAILURES: Final = 3
"""Identical failing calls (same tool, same arguments) before the run stops."""

MAX_CONSECUTIVE_FAILURES: Final = 10
"""Failing calls in a row, of any kind, before the run stops."""

MAX_TOTAL_FAILURES: Final = 20
"""Failing calls in all, before the run stops; catches failures between successes."""

REPEAT_NOTICE_AFTER: Final = 3
"""Identical calls in a row after which the result carries a notice to move on."""

MAX_REPEATED_CALLS: Final = 6
"""Identical calls in a row, failing or not, before the run stops."""

CONTEXT_TRIGGER_TOKENS: Final = 20_000
"""Estimated history size above which older tool outputs become a placeholder.

The middleware estimates 4 characters per token; the JSON tool traffic here
takes about 2 per token (Haiku 5.5 on a 21-storey building), so this is
about 40,000 real tokens. A 40,000 estimate never fired on that run, whose
prompts reached 73,770 tokens."""

_ARGUMENT_ECHO: Final = re.compile(
    r"^Error invoking tool '[^']+' with kwargs .*? with error:\s*", re.DOTALL
)


class FailureLoopGuard(AgentMiddleware):
    """Stop the agent run once tool calls keep failing or keep repeating.

    A model that cannot produce valid arguments tends to resend the same call
    indefinitely; LangGraph's default step limit is about ten thousand, so
    without this guard the loop only ends when the LLM budget does. Every
    failure is logged, and once tripped the guard ends this and any later run
    of the same agent with a message naming the failing call.

    A successful call can loop too: with a needed object missing, a model
    forced to call some tool re-listed the same objects 538 times. From the
    REPEAT_NOTICE_AFTER-th identical call in a row the result says it will not
    change; at MAX_REPEATED_CALLS the run stops.

    Argument errors from LangChain repeat the whole call before the field
    errors; the call is already in the model's own message, so the echo is cut
    to keep every retry from resending the arguments twice.
    """

    def __init__(self) -> None:
        super().__init__()
        self._failures: Counter[str] = Counter()
        self._consecutive = 0
        self._total = 0
        self._last_call: str | None = None
        self._repeats = 0
        self.reason: str | None = None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        call = request.tool_call
        key = f"{call['name']}({json.dumps(call['args'], sort_keys=True)})"
        self._repeats = self._repeats + 1 if key == self._last_call else 1
        self._last_call = key
        if result.status != "error":
            self._consecutive = 0
            return self._repeated(result, call["name"])
        result = result.model_copy(
            update={"content": _ARGUMENT_ECHO.sub("", str(result.content))}
        )
        self._failures[key] += 1
        self._consecutive += 1
        self._total += 1
        logger.warning(
            "Tool {} failed ({} identical, {} in a row): {}",
            call["name"],
            self._failures[key],
            self._consecutive,
            str(result.content)[:300],
        )
        if self._failures[key] >= MAX_REPEATED_FAILURES:
            self.reason = (
                f"{call['name']} failed {self._failures[key]} times with the "
                f"same arguments; last error: {result.content}"
            )
        elif self._consecutive >= MAX_CONSECUTIVE_FAILURES:
            self.reason = (
                f"{self._consecutive} tool calls failed in a row; "
                f"last error from {call['name']}: {result.content}"
            )
        elif self._total >= MAX_TOTAL_FAILURES:
            self.reason = (
                f"{self._total} tool calls failed in this phase; "
                f"last error from {call['name']}: {result.content}"
            )
        return result

    def _repeated(self, result: ToolMessage, name: str) -> ToolMessage:
        if self._repeats >= MAX_REPEATED_CALLS:
            self.reason = (
                f"{name} was called {self._repeats} times in a row with the "
                "same arguments"
            )
        if self._repeats < REPEAT_NOTICE_AFTER:
            return result
        logger.warning("Tool {} called {} times in a row", name, self._repeats)
        notice = (
            f"\n\nNOTE: call {self._repeats} in a row of {name} with the same "
            "arguments; the result will not change. Act on it, or give your "
            "final answer now and list in it what is missing."
        )
        return result.model_copy(update={"content": f"{result.content}{notice}"})

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        if self.reason is None:
            return None
        logger.error("Agent run stopped: {}", self.reason)
        return {
            "jump_to": "end",
            "messages": [AIMessage(content=f"Stopped: {self.reason}")],
        }


def build_agent(
    config: LLMConfig | None = None,
    system_prompt: str | None = None,
    tools: list[BaseTool] | None = None,
    response_format: type[BaseModel] | None = None,
    middleware: Sequence[AgentMiddleware] = (),
):
    """Build a tool-calling agent with optional structured final output.

    Args:
        config: Optional override. If None, reads src/configs/llm.yaml.
        system_prompt: Phase prompt; `language_directive()` is appended here
            so per-phase prompts stay free of language boilerplate.
        tools: Tools bound to the agent.
        response_format: Pydantic schema for the final structured answer,
            surfaced as `result["structured_response"]`.
        middleware: Extra middleware, e.g. `trace_middleware(collector)`.

    Returns:
        A compiled agent graph taking/returning `{"messages": [...]}`.
    """
    return create_agent(
        model=create_llm(config),
        tools=tools or [],
        system_prompt=(system_prompt or "") + language_directive(),
        # The final answer as a validated tool call, not free text. ToolStrategy
        # over the provider's JSON-schema mode: Claude's structured outputs
        # reject the idfpy schema keywords (note, units) that gateways pass on.
        response_format=ToolStrategy(response_format) if response_format else None,
        middleware=[
            FailureLoopGuard(),
            # Phase agents resend their whole history on every call; this caps
            # what large buildings and failure streaks add to each prompt.
            ContextEditingMiddleware(
                edits=[ClearToolUsesEdit(trigger=CONTEXT_TRIGGER_TOKENS, keep=5)]
            ),
            _sequential_tool_calls,
            serial_write_tools_middleware(),
            *middleware,
        ],
    )
