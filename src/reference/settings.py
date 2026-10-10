from pathlib import Path
from typing import Final, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LIBRARY: Final = Path("/tmp/ep-agent-reference/reference_library.sqlite")
"""Rebuilt or pulled from the hub on demand, so it lives in the system tmp."""


class ReferenceSettings(BaseSettings):
    """Where the reference library, its vectors and the embedding service live.

    Retrieval is on when both the Qdrant URL and the embedding URL are set,
    e.g. REFERENCE_QDRANT_URL=http://pan-office:6333 and
    REFERENCE_EMBEDDING_URL=http://pan-office:8000/v1.
    """

    model_config = SettingsConfigDict(
        env_prefix="REFERENCE_", env_file=".env", extra="ignore"
    )

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    collection: str = "ep_reference_qwen3_embedding_8b"
    embedding_url: str | None = None
    embedding_model: str = "qwen3-embedding-8b"
    embedding_api_key: SecretStr | None = None
    hub_repo: str = "ITOTII/ep-agent-rag"
    library_path: Path = DEFAULT_LIBRARY

    @model_validator(mode="after")
    def _both_or_neither(self) -> Self:
        if (self.qdrant_url is None) != (self.embedding_url is None):
            raise ValueError(
                "set both REFERENCE_QDRANT_URL and REFERENCE_EMBEDDING_URL, or neither"
            )
        return self

    @property
    def enabled(self) -> bool:
        return self.qdrant_url is not None
