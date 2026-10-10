import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from fastmcp import Client

from src.mcp.server import create_mcp_server

type CallTool = Callable[[str, dict[str, Any]], dict[str, Any]]


@pytest.fixture
def call_tool() -> CallTool:
    """Call a tool on a fresh server; returns the ToolResponse fields."""
    mcp = create_mcp_server()

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with Client(mcp) as client:
            return (await client.call_tool(name, args)).data["result"]

    return lambda name, args: asyncio.run(call(name, args))
