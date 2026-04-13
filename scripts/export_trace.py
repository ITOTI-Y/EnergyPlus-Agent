"""Run the demo agent and export every LangSmith trace to local JSON.

Uses `collect_runs` to capture the complete tree (every node / LLM call /
tool call with inputs, outputs, timing, errors) while the graph executes,
so the traces are available locally without a round-trip to the LangSmith
server. Files land in `output/traces/`.

Usage:
    uv run python scripts/export_trace.py

Requirements:
    - .env has LANGSMITH_API_KEY + LANGSMITH_TRACING=true (optional — the
      callback still collects runs locally even without tracing enabled).
    - data/dependencies/Energy+.idd present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers

from scripts._share import USER_INPUT
from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import auto_approval, print_final_messages, run_session


def _dump_traces(traced_runs: list[Any]) -> None:
    """Serialize every collected root run to output/traces/ as JSON."""
    traces_dir = Path("output/traces")
    traces_dir.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(traced_runs):
        path = traces_dir / f"run_{i:02d}_{run.id}.json"
        path.write_text(
            json.dumps(_run_to_dict(run), default=str, indent=2, ensure_ascii=False)
        )
        print(f"wrote {path}")


def _run_to_dict(run: Any) -> dict[str, Any]:
    """Recursively convert a langsmith Run tree to JSON-friendly dict."""
    return {
        "id": str(run.id),
        "name": run.name,
        "run_type": run.run_type,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "inputs": run.inputs,
        "outputs": run.outputs,
        "error": run.error,
        "tags": run.tags,
        "extra": run.extra,
        "children": [_run_to_dict(c) for c in (run.child_runs or [])],
    }


def main() -> None:
    epw = Path("data/weather/shenzhen/CHN_Guangdong.Shenzhen.594930_SWERA.epw")
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph()
    initial = AgentState(user_input=USER_INPUT)
    context = SimContext(epw_path=epw, output_dir=output_dir)
    cfg: RunnableConfig = {"configurable": {"thread_id": "export-demo"}}

    with collect_runs() as cb:
        try:
            state = run_session(
                graph, initial, context, cfg, on_interrupt=auto_approval
            )
            print_final_messages(state, n=3)
        finally:
            wait_for_all_tracers()
            _dump_traces(cb.traced_runs)


if __name__ == "__main__":
    main()
