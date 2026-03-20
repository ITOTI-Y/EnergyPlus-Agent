import sqlite3
import time
from datetime import datetime

from src.rag.chunk import Chunk, SQLiteProcessor
from src.rag.embedding import GeminiEmbeddingModel, GeminiTaskType
from src.rag.vector import QdrantVectorStore
from src.utils.logging import get_logger


class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_collection_name: str,
        gemini_api_key: str,
        index_db_path: str = "data/database/EP_Agent_data.db",
    ):
        self.sqlite_processor = SQLiteProcessor(db_path=index_db_path)
        self.vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection_name,
        )
        self.embedding_model = GeminiEmbeddingModel(api_key=gemini_api_key)
        self.logger = get_logger(__name__)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
        score_threshold: float | None = 0.5,
    ) -> list[dict]:
        embeddings = self.embedding_model.embed_batch(
            [query], task_type=GeminiTaskType.RETRIEVAL_QUERY
        )[0]
        results = self.vector_store.search(
            embeddings, top_k, chunk_type, score_threshold
        )
        return results

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
        score_threshold: float | None = 0.5,
    ) -> str:
        results = self.search(query, top_k, chunk_type, score_threshold)

        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results):
            content = result.get("data_description", "")
            table = result.get("vectored_table_name", "")
            record_id = result.get("record_id", "")
            score = result.get("score", 0.0)

            context_parts.append(
                f"[Document {i + 1}] (Table: {table}, RecordID: {record_id}, Score: {score:.2f})\nDescription: {content}\n---\n"
            )

        return "\n".join(context_parts)

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.embedding_model.embed_batch(texts)

    def chunk(
        self,
        table_name: str,
        data_id: int,
    ) -> Chunk:
        result = self.sqlite_processor.process_data(
            table_name=table_name, data_id=data_id
        )
        if result is None:
            self.logger.error(f"Failed to chunk {table_name}-{data_id}.")
            raise ValueError(f"Failed to chunk {table_name}-{data_id}.")
        return result

    def _get_all_chunks_table_id(self) -> list[dict]:
        all_points = self.vector_store.get_all_points()
        chunk_table_ids = []
        for point in all_points:
            chunk_table_ids.append(
                {
                    "table_name": point["vectored_table_name"],
                    "record_id": point["record_id"],
                    "data_datetime": point["datetime"],
                }
            )
        return chunk_table_ids

    def _get_all_sql(self) -> list[dict]:
        with sqlite3.connect(self.sqlite_processor.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            all_sql_records = []
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"SELECT id, datetime FROM [{table}]")
                for row in cursor.fetchall():
                    all_sql_records.append(
                        {
                            "table_name": table,
                            "record_id": row["id"],
                            "data_datetime": row["datetime"],
                        }
                    )
        return all_sql_records

    def check_rag_sync(self) -> list[dict]:
        vector_points = self._get_all_chunks_table_id()
        all_sql_records = self._get_all_sql()

        synced_set = {
            (d["table_name"], d["record_id"], d["data_datetime"]) for d in vector_points
        }

        unsync_data = []
        for record in all_sql_records:
            key = (record["table_name"], record["record_id"], record["data_datetime"])
            if key not in synced_set:
                unsync_data.append(record)

        return unsync_data

    def _embed_and_upsert(self, chunks: list[Chunk], batch_count: int = 100):
        if batch_count <= 0:
            raise ValueError("batch_count must be greater than 0")
        self.logger.info("--------Begin embedding-------")
        failed_batches = 0
        for i in range(0, len(chunks), batch_count):
            batch = chunks[i : i + batch_count]
            try:
                descriptions = [chunk.data_description for chunk in batch]
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.logger.info(
                    f"Embedding: {min(i + batch_count, len(chunks))}/{len(chunks)} [{now_str}]"
                )
                embeddings = self.embed(descriptions)
                self.vector_store.add(batch, embeddings)
                failed_batches = 0
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                self.logger.info(
                    f"Finish: {min(i + batch_count, len(chunks))}/{len(chunks)} [{now_str}]"
                )
            except Exception:
                failed_batches += 1
                self.logger.exception(f"Failed to process batch {i // batch_count}")
                if failed_batches >= 10:
                    self.logger.error("Too many consecutive failures, aborting.")
                    break
                time.sleep(1)
                continue
        if failed_batches:
            self.logger.warning(
                f"Embedding completed with {failed_batches} failed batch(es)."
            )

    def sync_rag(
        self,
        batch_count: int = 100,
    ):
        unsync_data = self.check_rag_sync()
        self.logger.info(f"Found {len(unsync_data)} records to vectorize.")
        chunks = []
        for record in unsync_data:
            table = record["table_name"]
            record_id = record["record_id"]
            self.logger.info(f"Chunking {table}-{record_id}")
            try:
                chunk = self.chunk(table, record_id)
            except ValueError:
                self.logger.error(f"Skipping {table}-{record_id}: chunk not found")
                continue
            chunks.append(chunk)
        self._embed_and_upsert(chunks, batch_count)

    def clean_zero_rag(
        self,
        batch_count: int = 100,
    ):
        zero_points = self.vector_store.get_zero_vector_points()
        self.logger.info(f"Found {len(zero_points)} records to re-vectorize.")
        chunks = []
        for record in zero_points:
            table = record["table_name"]
            record_id = record["record_id"]
            self.logger.info(f"Chunking {table}-{record_id}")
            try:
                chunk = self.chunk(table, record_id)
            except ValueError:
                self.logger.error(f"Skipping {table}-{record_id}: chunk not found")
                continue
            chunks.append(chunk)
        self._embed_and_upsert(chunks, batch_count)
