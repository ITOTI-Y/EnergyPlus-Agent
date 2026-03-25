from abc import ABC, abstractmethod
from typing import Any

from src.rag.chunk import Chunk
from src.utils.logging import get_logger

_PAYLOAD_CORE_KEYS = {
    "data_description",
    "vectored_table_name",
    "record_id",
    "data_dict",
    "datetime",
}


def _extract_payload(
    payload: dict[str, Any],
    *,
    point_id: Any = None,
    score: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data_description": payload.get("data_description", ""),
        "vectored_table_name": payload.get("vectored_table_name", ""),
        "record_id": payload.get("record_id", ""),
        "datetime": payload.get("datetime", 0),
        "full_data": payload.get("data_dict", {}),
        "metadata": {k: v for k, v in payload.items() if k not in _PAYLOAD_CORE_KEYS},
    }
    if point_id is not None:
        result["id"] = point_id
    if score is not None:
        result["score"] = score
    return result


class IVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: list[float],
        top_k: int,
        table_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
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

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )
        self.collection_name = collection_name
        self.dimension = dimension
        self.logger = get_logger(__name__)
        self._create_collection()

    def _create_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size  # type: ignore
            if existing_size != self.dimension:
                raise ValueError(
                    f"Collection '{self.collection_name}' exists with dimension {existing_size}, "
                    f"but expected {self.dimension}."
                )
            self.logger.info(
                f"Collection {self.collection_name} already exists (dimension={existing_size})"
            )

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 100,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=chunk.to_qdrant_payload(),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + batch_size],
            )

        self.logger.info(f"Added {len(chunks)} chunks to {self.collection_name}")

    def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),  # ty: ignore[invalid-argument-type]
        )
        self.logger.info(f"Deleted {len(ids)} points from {self.collection_name}")

    def search(
        self,
        query: list[float],
        top_k: int = 10,
        table_name: str | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        search_filter = None
        if table_name:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="vectored_table_name",
                        match=MatchValue(value=table_name),
                    )
                ]
            )

        query_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            query_filter=search_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        ).points

        return [
            _extract_payload(point.payload or {}, score=point.score)
            for point in query_results
        ]

    def get_all_points(self) -> list[dict]:
        all_results: list[dict] = []
        offset = None

        while True:
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )

            for record in scroll_result:
                all_results.append(
                    _extract_payload(record.payload or {}, point_id=record.id)
                )

            offset = next_offset
            if offset is None:
                break

        self.logger.info(
            f"Retrieved total {len(all_results)} points from {self.collection_name}"
        )
        return all_results

    def get_zero_vector_points(self) -> list[dict]:
        zero_points: list[dict] = []
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
                if record.vector is not None and all(v == 0.0 for v in record.vector):
                    payload = record.payload or {}
                    zero_points.append(
                        {
                            "id": record.id,
                            "table_name": payload.get("vectored_table_name", ""),
                            "record_id": payload.get("record_id", ""),
                            "datetime": payload.get("datetime", ""),
                        }
                    )

            offset = next_offset
            if offset is None:
                break

        self.logger.info(f"Scan complete. Found {len(zero_points)} zero vector(s).")
        return zero_points
