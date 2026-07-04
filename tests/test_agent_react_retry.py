from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.agent.react import ReactState, build_react_agent


@tool
def mark_done() -> str:
    """Mark the phase as inspected."""
    return "done"


class _FakeToolCallingModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, parallel_tool_calls=False):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(content="I am done without tools.")
        if len(self.calls) == 2:
            return AIMessage(
                content="",
                tool_calls=[{"name": "mark_done", "args": {}, "id": "call_1"}],
            )
        return AIMessage(content="finished after tool")


class _FakeFinishingModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, parallel_tool_calls=False):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="finished")


def test_react_retries_first_turn_without_tool_calls():
    model = _FakeToolCallingModel()
    agent = build_react_agent(
        llm=model,
        tools=[mark_done],
        system_prompt="Use tools.",
    )

    result = agent.invoke(ReactState(messages=[HumanMessage(content="run")]))

    assert len(model.calls) == 3
    assert any(
        "You have not called any tools yet" in getattr(m, "content", "")
        for m in result["messages"]
    )
    assert result["messages"][-1].content == "finished after tool"


def test_react_allows_text_finish_after_tool_history():
    model = _FakeFinishingModel()
    agent = build_react_agent(
        llm=model,
        tools=[mark_done],
        system_prompt="Use tools.",
    )

    state = ReactState(
        messages=[
            HumanMessage(content="run"),
            ToolMessage(content="done", tool_call_id="call_1"),
        ]
    )
    result = agent.invoke(state)

    assert len(model.calls) == 1
    assert result["messages"][-1].content == "finished"
