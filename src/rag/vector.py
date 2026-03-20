from abc import ABC, abstractmethod
from src.rag.chunk import Chunk

from src.utils.logging import get_logger


class IVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query: list[float], top_k: int, table_name: str | None = None, score_threshold: float | None = None) -> list[dict]:
        pass

    @abstractmethod
    def get_all_points(self) -> list[dict]:
        pass


class QdrantVectorStore(IVectorStore):
    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        prefer_grpc: bool = True,
        dimension: int = 3072,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )
        self.collection_name = collection_name
        self.distance = Distance
        self.vector_params = VectorParams
        self.dimension = dimension
        self.logger = get_logger(__name__)
        self._create_collection()

    def _create_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_params(
                    size=self.dimension,
                    distance=self.distance.COSINE,
                ),
            )
        else:
            self.logger.info(f"Collection {self.collection_name} already exists")

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(PointStruct(
                id=chunk.chunk_id, # type: ignore
                vector=embedding,
                payload=chunk.to_qdrant_payload(), # type: ignore
            ))

        for i in range(0, len(points), batch_size):
            batch_points = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
            )

        self.logger.info(f"Added {len(chunks)} chunks to {self.collection_name}")

    def search(
        self,
        query: list[float],
        top_k: int = 10,
        table_name: str | None = None,
        score_threshold: float | None = None,
    ):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = None
        if table_name:
            search_filter = Filter(
                must=[FieldCondition(
                    key="vectored_table_name",
                    match=MatchValue(value=table_name),
                )]
            )

        query_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        results = []  
        for result in query_results:  
            payload = result.payload or {}  
            results.append({  
                "data_description": payload.get("data_description", ""),  
                "vectored_table_name": payload.get("vectored_table_name", ""),  
                "record_id": payload.get("record_id", ""),  
                "full_data": payload.get("data_dict", ""),  
                "score": result.score,  
                "metadata": {  
                    k: v for k, v in payload.items()  
                    if k not in {"data_description", "vectored_table_name", "record_id", "data_dict", "datetime"}  
                },  
            }) 

        return results
    
    def get_all_points(self) -> list[dict]:
        all_results = []
        offset = None

        while True:
            
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False, 
                offset=offset,
            )

            for record in scroll_result:
                payload = record.payload or {}
                all_results.append({
                    "id": record.id,
                    "data_description": payload.get("data_description", ""),
                    "vectored_table_name": payload.get("vectored_table_name", ""),
                    "record_id": payload.get("record_id", ""),
                    "datetime": payload.get("datetime", 0),
                    "full_data": payload.get("data_dict", {}),
                    "metadata": {
                        k: v for k, v in payload.items() 
                        if k not in ["data_description", "vectored_table_name", "record_id", "data_dict"]
                    }
                })

            offset = next_offset
            
            if offset is None:
                break

        self.logger.info(f"Retrieved total {len(all_results)} points from {self.collection_name}")
        return all_results
    
    def get_zero_vector_points(self) -> list[dict]:
        import numpy as np
        zero_points = []
        offset = None

        self.logger.info(f"Scanning for zero vectors in {self.collection_name}...")

        while True:
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=True, 
                offset=offset,
                limit=100, 
            )

            for record in scroll_result:
                if record.vector is not None:
                    vec = np.array(record.vector)
   
                    if np.allclose(vec, 0):
                        payload = record.payload or {}
                        zero_points.append({
                            "id": record.id,
                            "table_name": payload.get("vectored_table_name", ""),
                            "record_id": payload.get("record_id", ""),
                            "datetime": payload.get("datetime", "")
                        })

            offset = next_offset
            if offset is None:
                break

        self.logger.info(f"Scan complete. Found {len(zero_points)} zero vector(s).")
        return zero_points
