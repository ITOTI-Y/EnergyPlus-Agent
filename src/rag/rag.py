from src.rag.chunk import Chunk, SQLiteProcessor
from src.rag.embedding import GeminiEmbeddingModel, GeminiTaskType
from src.rag.vector import QdrantVectorStore
from loguru import logger
from datetime import datetime
from pathlib import Path
import sqlite3
import time
class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        qdrant_collection_name: str,
        gemini_api_key: str,
    ):
        self.sqlite_processor = SQLiteProcessor()
        self.vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection_name,
        )
        self.embedding_model = GeminiEmbeddingModel(api_key=gemini_api_key)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: str | None = None,
        score_threshold: float | None = 0.5,
    ) -> list[dict]:
        embeddings = self.embedding_model.embed_batch([query],task_type=GeminiTaskType.RETRIEVAL_QUERY)[0]
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
        result = self.embedding_model.embed_batch(texts)
        return result
    
    def chunk(
        self,
        table_name: str,
        data_id: int,
    ) -> Chunk | None:
        ck = self.sqlite_processor.process_data(table_name=table_name, data_id=data_id)
        return ck
    
    def _get_all_chunks_table_id(self) -> list[dict]:
        aps = self.vector_store.get_all_points()
        cti = []
        for ap in aps:
            vtn = ap['vectored_table_name']
            rid = ap['record_id']
            dt = ap['datetime']
            cti.append({'table_name':vtn, 'record_id':rid, 'data_datetime':dt})
        return cti

    def _get_all_sql(self) -> list[dict]:
        with sqlite3.connect(self.sqlite_processor.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            all_sql = []
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"SELECT id, datetime FROM [{table}]")
                for row in cursor.fetchall():
                    all_sql.append({
                        'table_name': table,
                        'record_id': row['id'],
                        'data_datetime': row['datetime'],
                    })
        return all_sql

    def check_rag_sync(self) -> list[dict]:
        vp = self._get_all_chunks_table_id()
        asq = self._get_all_sql()

        vp_set = { (d['table_name'], d['record_id'], d['data_datetime']) for d in vp }

        unsync_data = []
        for sq in asq:
            sq_tuple = (sq['table_name'], sq['record_id'], sq['data_datetime'])
            if sq_tuple not in vp_set:
                unsync_data.append(sq)

        return unsync_data
        
    def _embed_and_upsert(self, cks: list[Chunk], batch_count: int = 100):
        print("--------Begin embedding-------")
        for i in range(0, len(cks), batch_count):
            time.sleep(1)
            batch = cks[i:i + batch_count]
            try:
                descriptions = [chunk.data_description for chunk in batch]
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(f"Embedding: {min(i + batch_count, len(cks))}/{len(cks)} [{now_str}]")
                embeddings = self.embed(descriptions)
                self.vector_store.add(batch, embeddings) # type: ignore
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                print(f"Finish: {min(i + batch_count, len(cks))}/{len(cks)} [{now_str}]")
            except Exception as e:
                print(f"Failed to process batch {i // batch_count}: {e}")
                continue

    def sync_rag(
            self,
            batch_count: int = 100,
    ):
        unsync_data = self.check_rag_sync()
        print(f"Find {len(unsync_data)} data needs vectorized.")
        cks = []
        for ud in unsync_data:
            print(f'chunking {ud['table_name']}-{ud["record_id"]}')
            ck = self.chunk(ud['table_name'], ud['record_id'])
            if ck is None:
                print(f'skipping {ud['table_name']}-{ud['record_id']}: chunk not found')
                continue
            cks.append(ck)
        self._embed_and_upsert(cks, batch_count)

    def clean_zero_rag(
            self,
            batch_count: int = 100,
    ):
        zero_points = self.vector_store.get_zero_vector_points()
        print(f"Find {len(zero_points)} data needs re_vectorized.")
        cks = []
        for ud in zero_points:
            print(f'chunking {ud['table_name']} - {ud['record_id']}')
            ck = self.chunk(ud['table_name'], ud['record_id'])
            if ck is None:
                print(f'skipping {ud['table_name']}-{ud['record_id']}: chunk not found')
                continue
            cks.append(ck)
        self._embed_and_upsert(cks, batch_count)