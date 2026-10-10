from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )

    provider: str = Field(..., description="The provider of the LLM model")
    base_url: str | None = Field(
        default=None, description="The base URL of the LLM model"
    )
    model_name: str = Field(..., description="The name of the LLM model to use")
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        description="Sampling temperature; None sends none, as Claude 5.5 "
        "models reject non-default values",
    )
    max_tokens: int = Field(
        ..., ge=0, description="The maximum number of tokens to generate"
    )
    api_key: str | None = Field(
        default=None, description="The API key of the LLM model"
    )
    reasoning_max_tokens: int | None = Field(
        default=None,
        gt=0,
        description="Thinking budget, sent as OpenRouter's reasoning.max_tokens; "
        "None sends none",
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
