from langchain_openai import ChatOpenAI

from src.agent.llm import create_llm
from src.configs.config import LLMConfig


def _config(**overrides) -> LLMConfig:
    defaults = {
        "provider": "openai",
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 1000,
        "api_key": "test-key",
    }
    return LLMConfig.model_validate({**defaults, **overrides})


def test_create_llm_sets_default_thinking_budget():
    llm = create_llm(_config())
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"max_thinking_budget": 4096}


def test_create_llm_respects_configured_thinking_budget():
    llm = create_llm(_config(max_thinking_budget=8192))
    assert isinstance(llm, ChatOpenAI)
    assert llm.extra_body == {"max_thinking_budget": 8192}
