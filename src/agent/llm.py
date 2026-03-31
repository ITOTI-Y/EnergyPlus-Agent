from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from omegaconf import OmegaConf

from src.configs.config import LLMConfig

load_dotenv()


def create_llm(config: LLMConfig | None = None) -> BaseChatModel:
    if config is None:
        dict_config = OmegaConf.load(
            Path(__file__).parent.parent / "configs" / "llm.yaml"
        )
        config = LLMConfig.model_validate(dict_config)
    if config.provider == "openai":
        return ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_base=config.base_url,
        )
    elif config.provider == "anthropic":
        return ChatAnthropic(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            anthropic_api_url=config.base_url,
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


llm = create_llm()
