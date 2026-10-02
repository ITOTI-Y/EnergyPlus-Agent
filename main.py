import os
import time
from pathlib import Path
from typing import Annotated, Literal

import typer
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from typer import Argument, Option

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import interactive_approval, print_final_messages, run_session
from src.utils.logging import get_logger, setup_logger

load_dotenv()

logger_time = time.strftime("%Y%m%d_%H%M%S")
setup_logger(
    level="INFO",
    console_output=True,
    log_file_path=Path(f"./output/logs/{logger_time}.log"),
)
logger = get_logger(__name__)

app = typer.Typer()


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


@app.command()
def run_agent(
    user_input: Annotated[str, Argument(help="Natural language building description")],
    epw: Annotated[Path, Option("--epw", "-w", help="Path to the EPW weather file")],
    images: Annotated[
        list[Path],
        Option(
            "--image",
            "-i",
            default_factory=list,
            show_default=False,
            help="Architectural drawing(s); repeat flag for multiple (floorplan + elevation + perspective...)",
        ),
    ],
    output_dir: Annotated[
        Path,
        Option(
            "--output-dir",
            "-o",
            help="Output directory for EnergyPlus simulation results",
        ),
    ] = Path("output"),
    thread_id: Annotated[
        str,
        Option(
            "--thread-id",
            "-t",
            help="Unique identifier for this conversation thread",
        ),
    ] = "demo",
) -> None:
    """Run the multi-phase agent end-to-end.

    Stops at the validate interrupt; print the pending summary and loop
    until the user types 'approve' or feedback text.
    """
    graph = build_graph()
    initial = AgentState(
        user_input=user_input,
        image_paths=[str(p) for p in images],
    )
    context = SimContext(epw_path=epw, output_dir=output_dir)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    state = run_session(
        graph, initial, context, config, on_interrupt=interactive_approval
    )
    print_final_messages(state)


if __name__ == "__main__":
    app()
