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


reference_app = typer.Typer(
    help="Build, index and share the prototype reference library"
)
app.add_typer(reference_app, name="reference")


@reference_app.command("build")
def reference_build(
    downloads: Annotated[
        Path, Option(help="Directory for the downloaded prototype zips")
    ] = Path("/tmp/ep-agent-prototypes/raw_doe_prototypes"),
) -> None:
    """Download the DOE prototypes and build the library from them."""
    import tempfile

    from src.reference import library, prototypes
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    archives = prototypes.download(prototypes.archive_urls(), downloads)
    with tempfile.TemporaryDirectory(prefix="ep-prototypes-") as work:
        models = prototypes.extract(archives, Path(work))
        report = library.build(
            models, settings.library_path, {"source": "energycodes.gov"}
        )
    logger.info(
        "Built {} from {} models: {}; {} schedules skipped",
        settings.library_path,
        report.models,
        report.entries,
        len(report.skipped_schedules),
    )


@reference_app.command("embed")
def reference_embed() -> None:
    """Embed library entries that have no embedding yet."""
    from src.reference.index import Embedder, embed_library
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    if settings.embedding_url is None:
        raise typer.BadParameter("set REFERENCE_EMBEDDING_URL")
    embedder = Embedder(settings.embedding_url, settings.embedding_model)
    count = embed_library(settings.library_path, embedder)
    logger.info("Embedded {} entries", count)


@reference_app.command("load")
def reference_load() -> None:
    """Replace the Qdrant collection with the library's embedded entries."""
    from qdrant_client import QdrantClient

    from src.reference.index import load_collection
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    if settings.qdrant_url is None:
        raise typer.BadParameter("set REFERENCE_QDRANT_URL")
    key = settings.qdrant_api_key
    client = QdrantClient(
        url=settings.qdrant_url, api_key=key.get_secret_value() if key else None
    )
    count = load_collection(settings.library_path, client, settings.collection)
    logger.info("Loaded {} points into {}", count, settings.collection)


@reference_app.command("publish")
def reference_publish() -> None:
    """Upload the library to the private Hugging Face dataset repo."""
    from src.reference.hub import publish
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    logger.info("Published: {}", publish(settings.library_path, settings.hub_repo))


@reference_app.command("pull")
def reference_pull() -> None:
    """Download the library from the Hugging Face dataset repo."""
    from src.reference.hub import pull
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    logger.info("Pulled {}", pull(settings.hub_repo, settings.library_path))


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
    from src.reference.search import ReferenceSearch
    from src.reference.settings import ReferenceSettings

    settings = ReferenceSettings()
    # Configured but unreachable or mismatched fails here, before any LLM call.
    reference = ReferenceSearch.connect(settings, epw) if settings.enabled else None
    context = SimContext(epw_path=epw, output_dir=output_dir, reference=reference)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    state = run_session(
        graph, initial, context, config, on_interrupt=interactive_approval
    )
    print_final_messages(state)


if __name__ == "__main__":
    app()
