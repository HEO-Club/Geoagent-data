"""本机 stdio MCP：把 Canonical Tool 暴露给本地 Agent。

启动：``python -m tool.mcp``（工作目录为仓库根）。Cursor 配置示例：

{
  "mcpServers": {
    "geoagent-tools": {
      "command": ".venv/Scripts/python.exe",
      "args": ["-m", "tool.mcp"]
    }
  }
}
"""

from __future__ import annotations

from tool.mcp.catalog import McpToolSpec, load_mcp_tool_specs, mcp_tool_name
from tool.mcp.session import McpSession, McpSessionError

__all__ = [
    "McpSession",
    "McpSessionError",
    "McpToolSpec",
    "load_mcp_tool_specs",
    "mcp_tool_name",
]
