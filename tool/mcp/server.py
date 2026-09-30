"""stdio MCP 服务：把目录中的 operation 暴露给本地 Agent。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from tool.mcp.session import McpSession

SERVER_NAME = "geoagent-tools"


def create_server(session: McpSession | None = None) -> Server:
    """创建 stdio MCP Server；未传入会话时按环境变量初始化。"""

    bound = session or McpSession.from_env()

    async def on_list_tools(
        _ctx: Any,
        _params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=spec.name,
                    description=spec.description,
                    input_schema=spec.input_schema,
                )
                for spec in bound.list_specs()
            ]
        )

    async def on_call_tool(
        _ctx: Any,
        params: CallToolRequestParams,
    ) -> CallToolResult:
        arguments = params.arguments if isinstance(params.arguments, dict) else {}
        payload = bound.call(params.name, arguments)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(payload, ensure_ascii=False),
                )
            ]
        )

    return Server(
        SERVER_NAME,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run_stdio(session: McpSession | None = None) -> None:
    """以标准输入输出运行 MCP 服务，直到宿主关闭连接。"""

    app = create_server(session)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main() -> None:
    """``python -m tool.mcp`` 入口。"""

    asyncio.run(run_stdio())
