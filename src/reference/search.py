from dataclasses import dataclass
from pathlib import Path
from typing import Final

from loguru import logger
from openai import APIError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from src.reference.climate import Climate, from_epw
from src.reference.extract import Kind
from src.reference.index import Embedder, Hit, search
from src.reference.settings import ReferenceSettings

QDRANT_TIMEOUT_S: Final = 30
"""Per-request timeout. The client's 5 s default failed whole agent runs
when the link to the server dropped to about 300 KB/s with 1 s connects: a
search sends a 4096-value query vector as JSON."""


SEARCH_ATTEMPTS: Final = 2
"""A failed search is tried once more, then stops the run."""


class ReferenceSearchError(RuntimeError):
    """The reference library could not be searched.

    The phases take their values from the library where the specification
    gives none, so the run stops with the reason rather than continue on
    invented values.
    """


@dataclass(frozen=True, slots=True)
class ReferenceSearch:
    """Semantic search over the reference library for one building site."""

    client: QdrantClient
    collection: str
    embedder: Embedder
    climate: Climate
    qdrant_url: str = ""
    embedding_url: str = ""

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
            timeout=QDRANT_TIMEOUT_S,
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
        return cls(
            client,
            settings.collection,
            embedder,
            from_epw(epw_path),
            settings.qdrant_url,
            settings.embedding_url,
        )

    def find(
        self, query: str, kind: Kind, *, by_climate: bool, limit: int = 5
    ) -> list[Hit]:
        """Library objects of ``kind`` closest to ``query``.

        Raises:
            ReferenceSearchError: If the embedding service or Qdrant fails
                twice, naming the service, its URL and the search.
        """
        zones = self.climate.candidates if by_climate else None
        for attempt in range(1, SEARCH_ATTEMPTS + 1):
            try:
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
            except (APIError, ApiException) as e:
                service, url = (
                    ("embedding service", self.embedding_url)
                    if isinstance(e, APIError)
                    else ("Qdrant", self.qdrant_url)
                )
                problem = f"{service} at {url or '?'}: {type(e).__name__}: {e}"
                if attempt == SEARCH_ATTEMPTS:
                    raise ReferenceSearchError(
                        f"Reference search for {kind}s ({query!r}) failed "
                        f"{SEARCH_ATTEMPTS} times; {problem}"
                    ) from e
                logger.warning("Reference search failed, retrying: {}", problem)
        raise AssertionError("unreachable")
