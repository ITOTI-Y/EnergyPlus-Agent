from abc import ABC, abstractmethod
from enum import Enum

from dotenv import load_dotenv

from src.utils.logging import get_logger

load_dotenv()


class IEmbeddingModel(ABC):
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass


class GeminiTaskType(Enum):
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"
    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CODE_RETRIEVAL_QUERY = "CODE_RETRIEVAL_QUERY"
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    FACT_VERIFICATION = "FACT_VERIFICATION"


class GeminiEmbeddingModel(IEmbeddingModel):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-embedding-001",
        dimension: int = 3072,
    ):
        from google.genai import Client

        self.client: Client = Client(api_key=api_key)
        self.model_name = model_name
        self.dimension = dimension
        self.logger = get_logger(__name__)

    def embed_batch(
        self,
        texts: list[str],
        task_type: GeminiTaskType = GeminiTaskType.RETRIEVAL_DOCUMENT,
    ) -> list[list[float]]:
        from google.genai import types

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type.value,
                    output_dimensionality=self.dimension,
                ),
            )

        except Exception as e:
            self.logger.exception(f"Embedding API error: {e}")
            raise ValueError(f"Embedding API call failed: {e}") from e

        if not result.embeddings:
            raise ValueError("Embedding API returned no embeddings")

        if len(result.embeddings) != len(texts):
            raise ValueError(
                f"Embedding API returned {len(result.embeddings)} embeddings for {len(texts)} texts"
            )

        vectors: list[list[float]] = []
        for idx, emb in enumerate(result.embeddings):
            if emb.values is None:
                raise ValueError(
                    f"Embedding API returned an empty embedding at index {idx}"
                )
            vectors.append(emb.values)
        return vectors
