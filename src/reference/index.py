"""Embeddings and the Qdrant collection of the reference library.

Qwen3-Embedding expects an instruction before each query and none before
documents. The collection payload holds each object's full data, so a
search needs only Qdrant and the embedding service.
"""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import Any, Final

from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.reference import library
from src.reference.extract import Kind

QUERY_INSTRUCTION: Final = (
    "Given a building's location, type and the element it needs, retrieve "
    "EnergyPlus objects from DOE prototype buildings that suit it"
)
BATCH_SIZE: Final = 64


class Embedder:
    """An OpenAI-compatible embedding endpoint, e.g. vLLM serving Qwen3."""

    def __init__(self, base_url: str, model: str, api_key: str = "unused") -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def documents(self, texts: list[str]) -> list[list[float]]:
        reply = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in reply.data]

    def query(self, text: str) -> list[float]:
        return self.documents([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"])[0]


def embed_library(path: Path, embedder: Embedder) -> int:
    """Embed the library entries that have no embedding yet.

    Returns:
        The number of entries embedded.
    """
    done = {key for key, _ in library.embeddings(path)}
    pending = [e for e in library.entries(path) if e.id not in done]
    for batch in batched(pending, BATCH_SIZE):
        vectors = embedder.documents([e.description for e in batch])
        library.store_embeddings(
            path, {e.id: v for e, v in zip(batch, vectors, strict=True)}
        )
        logger.info("Embedded {} entries", len(batch))
    library.set_metadata(path, {"embedding_model": embedder.model})
    return len(pending)


def point_id(entry_id: str) -> str:
    return str(uuid.UUID(hex=entry_id))


def load_collection(path: Path, client: QdrantClient, collection: str) -> int:
    """Replace the collection with the library's embedded entries.

    Returns:
        The number of points written.
    """
    vectors = dict(library.embeddings(path))
    if not vectors:
        raise ValueError(f"{path} holds no embeddings; embed the library first")
    size = len(next(iter(vectors.values())))
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection, vectors_config=VectorParams(size=size, distance=Distance.COSINE)
    )
    points = (
        PointStruct(
            id=point_id(e.id),
            vector=vectors[e.id],
            payload={
                "id": e.id,
                "kind": e.kind,
                "object_type": e.object_type,
                "name": e.name,
                "data": e.data,
                "description": e.description,
                "categories": e.categories,
                "building_types": e.building_types,
                "standards": e.standards,
                "climate_zones": e.climate_zones,
                "roles": e.roles,
            },
        )
        for e in library.entries(path)
        if e.id in vectors
    )
    written = 0
    for batch in batched(points, 256):
        client.upsert(collection, points=list(batch))
        written += len(batch)
    return written


@dataclass(frozen=True, slots=True)
class Hit:
    score: float
    kind: Kind
    object_type: str
    name: str
    data: dict[str, Any]
    climate_zones: list[str]
    building_types: list[str]
    standards: list[str]
    roles: list[str]


def search(
    client: QdrantClient,
    collection: str,
    embedder: Embedder,
    query: str,
    *,
    kind: Kind,
    climate_zones: list[str] | None = None,
    limit: int = 5,
) -> Iterator[Hit]:
    """Entries of one kind most similar to the query, optionally by zone."""
    conditions = [FieldCondition(key="kind", match=MatchValue(value=kind))]
    if climate_zones:
        conditions.append(
            FieldCondition(key="climate_zones", match=MatchAny(any=climate_zones))
        )
    result = client.query_points(
        collection,
        query=embedder.query(query),
        query_filter=Filter(must=conditions),
        limit=limit,
        with_payload=True,
    )
    for point in result.points:
        payload = point.payload or {}
        yield Hit(
            point.score,
            payload["kind"],
            payload["object_type"],
            payload["name"],
            payload["data"],
            payload["climate_zones"],
            payload["building_types"],
            payload["standards"],
            payload["roles"],
        )
