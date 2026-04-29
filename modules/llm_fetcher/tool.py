import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Tool:
    """A single tool that an Agent can call."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable[..., Any]  # sync or async callable

    async def execute(self, **kwargs: Any) -> Any:
        """Invoke the tool handler, awaiting if necessary."""
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        return self.handler(**kwargs)


class ToolRegistry:
    """Registers and executes tools, and produces LLM-compatible schemas."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a tool. Returns the tool for decorator usage."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        """Retrieve a registered tool by name."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool by name."""
        tool = self.get(name)
        return await tool.execute(**arguments)

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-compatible 'tools' list for LLM requests."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_prompt_hint(self) -> str:
        """Return a prompt snippet that instructs the LLM how to call tools."""
        if not self._tools:
            return ""

        lines: List[str] = [
            "",
            "=== AVAILABLE TOOLS ===",
            "To call a tool, emit a JSON object in your response with exactly these two fields:",
            '  "tool": "<tool_name>",',
            '  "arguments": {<key>: <value>, ...}',
            ""
        ]
        for t in self._tools.values():
            lines.append(f"Tool: {t.name}")
            lines.append(f"  Description: {t.description}")
            params = json.dumps(t.parameters, ensure_ascii=False)
            lines.append(f"  Parameters: {params}")
            lines.append("")
        lines.append("=== END TOOLS ===")
        return "\n".join(lines)
