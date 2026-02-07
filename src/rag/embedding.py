from abc import ABC, abstractmethod
from enum import Enum
from time import sleep
from datetime import datetime
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
        self.api_key = api_key
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
                print(f"Batch returned no embeddings. Filling with zeros.")
                current_embeddings = [[0.0] * self.dimension for _ in batch]
            else:
                current_embeddings = []
                for embedding in result.embeddings:
                    vals = embedding.values
                        
                    if self.dimension < 3072: 
                        emb_np = np.array(vals)
                        norm = np.linalg.norm(emb_np)
                        if norm > 0:
                            emb_np = emb_np / norm
                        current_embeddings.append(emb_np.tolist())
                    else:
                        current_embeddings.append(vals)
                            
                
            all_values.extend(current_embeddings)

        except Exception as e:
            print(f"Embedding API error at batch: {e}")
                
            all_values.extend([[0.0] * self.dimension for _ in batch])

        return all_values