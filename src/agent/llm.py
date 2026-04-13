from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from omegaconf import OmegaConf

from src.configs.config import LLMConfig

load_dotenv()


def create_llm(config: LLMConfig | None = None) -> BaseChatModel:
    """Create a LangChain chat model from LLMConfig.

    Args:
        config: Optional override. If None, reads src/configs/llm.yaml.

    Returns:
        A BaseChatModel routed to the configured provider.
    """
    if config is None:
        raw = OmegaConf.load(
            Path(__file__).resolve().parent.parent / "configs" / "llm.yaml"
        )
        config = LLMConfig.model_validate(OmegaConf.to_container(raw, resolve=True))

    model_id = f"{config.provider}:{config.model_name}"
    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    return init_chat_model(model_id, **kwargs)
