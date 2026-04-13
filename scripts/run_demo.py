from pathlib import Path

from langchain_core.runnables import RunnableConfig

from scripts._share import USER_INPUT
from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import auto_approval, print_final_messages, run_session


def main() -> None:
    epw = Path("data/weather/Shenzhen.epw")
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    initial = AgentState(user_input=USER_INPUT)
    context = SimContext(epw_path=epw, output_dir=output_dir)
    cfg: RunnableConfig = {"configurable": {"thread_id": "demo"}}

    state = run_session(graph, initial, context, cfg, on_interrupt=auto_approval)
    print_final_messages(state, n=3)


if __name__ == "__main__":
    main()
