import sqlite3
import uuid

from pydantic import BaseModel, ConfigDict, Field

from src.utils.logging import get_logger


class Chunk(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )
    vectored_table_name: str = Field(
        description="The name of the table the chunk is vectored from"
    )
    record_id: int = Field(description="The ID of the record in the table")
    data_description: str = Field(description="The description content of the chunk")
    data_dict: dict = Field(description="The full data record as a dictionary")
    datetime: int = Field(description="The datetime when the chunk was created")
    metadata: dict = Field(
        default_factory=dict,
        description="The metadata of the chunk",
    )

    @property
    def chunk_id(self) -> str:
        content_str = f"energyplus_database_{self.vectored_table_name}_{self.record_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content_str))

    def to_qdrant_payload(self) -> dict:
        return {
            "data_description": self.data_description,
            "vectored_table_name": self.vectored_table_name,
            "record_id": self.record_id,
            "data_dict": self.data_dict,
            "datetime": self.datetime,
            **self.metadata,
        }


class SQLiteProcessor:
    def __init__(self, db_path: str = "data/database/EP_Agent_data.db"):
        self.db_path = db_path
        self.logger = get_logger(__name__)

    def process_data(
        self, table_name: str, data_id: int, content_column: str = "description"
    ) -> Chunk | None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                # Validate table_name to prevent SQL injection
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                )
                if not cursor.fetchone():
                    self.logger.error(f"Table {table_name} does not exist")
                    return None
                cursor.execute(f"SELECT * FROM [{table_name}] WHERE id = ?", (data_id,))
                result = cursor.fetchone()

                if result is None:
                    return None

                full_dict = dict(result)
                if content_column not in full_dict:
                    self.logger.error(
                        f"Column '{content_column}' not found in table {table_name}"
                    )
                    return None
                for required_col in ("id", "datetime"):
                    if required_col not in full_dict:
                        self.logger.error(
                            f"Required column '{required_col}' not found in table {table_name}"
                        )
                        return None
                exclude_keys = {content_column, "id", "datetime"}
                clean_data = {
                    k: v for k, v in full_dict.items() if k not in exclude_keys
                }

                return Chunk(
                    vectored_table_name=table_name,
                    record_id=data_id,
                    data_description=full_dict[content_column],
                    data_dict=clean_data,
                    datetime=full_dict["datetime"],
                )
        except Exception as e:
            self.logger.error(f"Error processing {table_name} ID {data_id}: {e}")
            raise
