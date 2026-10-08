from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from src.reference.climate import Climate, from_epw
from src.reference.extract import Kind
from src.reference.index import Embedder, Hit, search
from src.reference.settings import ReferenceSettings


@dataclass(frozen=True, slots=True)
class ReferenceSearch:
    """Semantic search over the reference library for one building site."""

    client: QdrantClient
    collection: str
    embedder: Embedder
    climate: Climate

    @classmethod
    def connect(cls, settings: ReferenceSettings, epw_path: Path) -> "ReferenceSearch":
        """Connect and check that the collection and the embedder agree.

        Raises:
            RuntimeError: If the collection is missing or its vectors do not
                match the embedding model's.
            qdrant_client.http.exceptions.ResponseHandlingException,
            openai.APIConnectionError: If a service is unreachable.
        """
        if settings.qdrant_url is None or settings.embedding_url is None:
            raise ValueError("reference search is not configured")
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key
            else None,
        )
        if not client.collection_exists(settings.collection):
            raise RuntimeError(
                f"Qdrant collection '{settings.collection}' not found; load it "
                "with `main.py reference load`"
            )
        embedder = Embedder(
            settings.embedding_url,
            settings.embedding_model,
            settings.embedding_api_key.get_secret_value()
            if settings.embedding_api_key
            else "unused",
        )
        vectors = client.get_collection(settings.collection).config.params.vectors
        expected = getattr(vectors, "size", None)
        actual = len(embedder.query("probe"))
        if expected != actual:
            raise RuntimeError(
                f"collection vectors have {expected} dimensions, "
                f"{settings.embedding_model} gives {actual}"
            )
        return cls(client, settings.collection, embedder, from_epw(epw_path))

    def find(
        self, query: str, kind: Kind, *, by_climate: bool, limit: int = 5
    ) -> list[Hit]:
        zones = self.climate.candidates if by_climate else None
        return list(
            search(
                self.client,
                self.collection,
                self.embedder,
                query,
                kind=kind,
                climate_zones=zones,
                limit=limit,
            )
        )
