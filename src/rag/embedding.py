from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
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

        all_values = []
        batch = texts

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch, # type: ignore
                config=types.EmbedContentConfig(
                    task_type=task_type.value,
                    output_dimensionality=self.dimension,
                ),
            )
                
            if not result.embeddings:
                self.logger.error("Batch returned no embeddings. Filling with zeros.")
                raise ValueError('This embedding dimension is not 3072, there is something wrong about embedding')
            else:
                current_embeddings = []
                for embedding in result.embeddings:
                    vals = embedding.values
                        
                    if self.dimension < 3072: 
                        self.logger.error('This embedding dimension is not 3072')
                        raise ValueError('This embedding dimension is not 3072, there is something wrong about embedding')
                    else:
                        current_embeddings.append(vals)
                            
            all_values.extend(current_embeddings)

        except Exception as e:
            self.logger.error(f"Embedding API error at batch: {e}")
            raise ValueError('This embedding dimension is not 3072, there is something wrong about embedding')

        return all_values