from pathlib import Path

from langchain_core.runnables import RunnableConfig
from loguru import logger
from typer import Exit

from scripts._share import HARD_USER_INPUT
from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import auto_approval, print_final_messages, run_session
from src.reference.search import ReferenceSearch, ReferenceSearchError
from src.reference.settings import ReferenceSettings
from src.utils.logging import setup_logger

setup_logger(level="INFO")


def main() -> None:
    epw = Path("data/weather/Shenzhen.epw")
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    settings = ReferenceSettings()
    reference = ReferenceSearch.connect(settings, epw) if settings.enabled else None
    initial = AgentState(user_input=HARD_USER_INPUT)
    context = SimContext(epw_path=epw, output_dir=output_dir, reference=reference)
    config: RunnableConfig = {"configurable": {"thread_id": "demo"}}

    try:
        state = run_session(graph, initial, context, config, on_interrupt=auto_approval)
    except ReferenceSearchError as e:
        logger.exception("Run stopped")
        raise Exit(1) from e
    print_final_messages(state)


if __name__ == "__main__":
    main()
