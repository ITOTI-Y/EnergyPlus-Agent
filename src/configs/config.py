from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    model_name: str = Field(default="gemini-embedding-001")
    dimension: int = Field(default=3072)
    task_type: str = Field(default="RETRIEVAL_DOCUMENT")

    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="forbid",
    )

    def __post_init__(self):
        with open(Path("configs/embedding.yaml")) as f:
            config = OmegaConf.load(f)
        self.model_name = config.model_name
        self.dimension = config.dimension
        self.task_type = config.task_type
