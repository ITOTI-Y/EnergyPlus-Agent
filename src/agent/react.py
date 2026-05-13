from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from src.agent._share import language_directive
from src.agent.trace import TraceCollector


class ReactState(BaseModel):
    """Internal state for phase-level ReAct subgraph.

    Kept separate from AgentState: messages here are NOT propagated
    to the outer graph. They exist only for the LLM <-> Tool loop.
    """

    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


def build_react_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    trace_collector: TraceCollector | None = None,
) -> CompiledStateGraph:
    """Build a 3-node ReAct subgraph.

    Topology: llm -> [tools_condition] -> tools -> llm -> ... -> END.
    `parallel_tool_calls=False` is enforced so each tool call can be
    validated sequentially — important because tool calls mutate
    the shared (local-copy) ConfigState.

    No checkpointer, no interrupts inside the subgraph.
    """
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    # Project-wide language directive appended once; per-phase prompts
    # stay free of language boilerplate.
    effective_prompt = system_prompt + language_directive()

    def llm_node(state: ReactState) -> dict:
        messages = [SystemMessage(content=effective_prompt), *state.messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
        wrap_tool_call=trace_collector.wrap if trace_collector else None,
    )

    builder = StateGraph(ReactState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile()
