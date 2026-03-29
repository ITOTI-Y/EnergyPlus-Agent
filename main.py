import os
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from dotenv import load_dotenv

from src.converter_manager import ConverterManager
from src.runner.runner import EnergyPlusRunner
from src.utils.logging import get_logger, setup_logger
from src.validator.data_model import BaseSchema

load_dotenv()

logger_time = time.strftime("%Y%m%d_%H%M%S")
setup_logger(
    level="INFO",
    console_output=True,
    log_file_path=Path(f"./output/logs/{logger_time}.log"),
)
logger = get_logger(__name__)

app = typer.Typer()

idd_file = Path("./data/dependencies/Energy+.idd")
BaseSchema.set_idf(idd_file)


@app.command()
def convert_idf():
    yaml_file = Path("./data/schemas/building_schema.yaml")
    idf_file_output = Path(f"./output/idf/output_{logger_time}.idf")
    epw_file = Path("./data/weather/Shenzhen.epw")
    manager = ConverterManager(yaml_file)
    manager.convert_all()
    manager.save_idf(idf_file_output)
    ep_runner = EnergyPlusRunner(manager.idf)
    ep_runner.run_idf(epw_file_path=epw_file)


@app.command()
def mcp_server(
    transport: Literal["stdio", "http", "sse", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    from src.mcp.server import mcp

    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, port=port, host=host)


@app.command()
def embedding(
    qdrant_collection_name: Annotated[
        str,
        typer.Option("--collection", "-c", help="The name of the Qdrant collection"),
    ],
    index_db_path: Annotated[
        str, typer.Option("--db-path", "-d", help="The path to the index database")
    ],
):
    import asyncio

    from src.rag.rag import RAGSystem

    qdrant_url = os.getenv("QDRANT_ENDPOINT")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not qdrant_url or not qdrant_api_key or not gemini_api_key:
        raise ValueError(
            "QDRANT_ENDPOINT, QDRANT_API_KEY, and GEMINI_API_KEY must be set"
        )
    rag_system = RAGSystem(
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        qdrant_collection_name=qdrant_collection_name,
        gemini_api_key=gemini_api_key,
        index_db_path=index_db_path,
    )
    result = asyncio.run(rag_system.sync_rag_async())
    if result.failed_count > 0:
        logger.error(f"Failed to embed {result.failed_count} batches")
        raise typer.Exit(1)
    logger.info(f"Successfully embedded {result.success_count} batches")


if __name__ == "__main__":
    app()
