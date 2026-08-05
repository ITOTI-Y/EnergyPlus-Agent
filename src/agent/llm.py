from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from omegaconf import OmegaConf
from pydantic import BaseModel

from src.agent._share import language_directive
from src.configs.config import LLMConfig

load_dotenv()


def _load_config() -> LLMConfig:
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
        config = _load_config()

    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "extra_body": {"max_thinking_budget": config.max_thinking_budget or 4096},
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return init_chat_model(config.model_name, model_provider=config.provider, **kwargs)


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
        response_format=response_format,
        middleware=[_sequential_tool_calls, *middleware],
    )
