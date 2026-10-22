from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Agent LLM settings, read from ``LLM_*`` environment variables.

    ``LLM_TEMPERATURE=null`` sends no temperature, for providers that reject
    one. The thinking budget is sent only when ``LLM_REASONING_MAX_TOKENS``
    is set, as the gateway extension ``reasoning.max_tokens``.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_parse_none_str="null",
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
        frozen=True,
    )

    provider: str = Field(
        default="openai",
        description="Provider name passed to LangChain init_chat_model",
    )
    base_url: str | None = Field(
        default=None, description="The base URL of the LLM model"
    )
    model_name: str = Field(
        validation_alias="LLM_MODEL", description="The name of the LLM model to use"
    )
    vision_model_name: str | None = Field(
        default=None,
        validation_alias="LLM_VISION_MODEL",
        description="Model that reads input images; without one, images are refused",
    )
    temperature: float | None = Field(
        default=0.7,
        ge=0.0,
        description="Sampling temperature; None sends none, for providers "
        "that reject one",
    )
    max_tokens: int = Field(
        default=16384, ge=0, description="The maximum number of tokens to generate"
    )
    api_key: str | None = Field(
        default=None, description="The API key of the LLM model"
    )
    reasoning_max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Thinking budget, sent as reasoning.max_tokens; None sends none",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Retries the provider performs on transient API errors",
    )
    timeout: float = Field(
        default=120.0,
        gt=0,
        description="Seconds before a request is abandoned and retried",
    )
