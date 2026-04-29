"""Expose ThinkingGraph operations as Tools so Agents can call them via tool_call."""

from typing import Any, Dict, List, Optional

from .thinking_graph import ThinkingEdgeType, ThinkingGraph, ThinkingNodeType
from .tool import Tool


def create_thinking_graph_tools(graph: ThinkingGraph) -> List[Tool]:
    """
    将 ThinkingGraph 的核心操作封装为一组 Tool。
    Agent 可以通过 tool_call 间接操作图，无需直接耦合 ThinkingGraph。

    Args:
        graph: 要操作的 ThinkingGraph 实例。

    Returns:
        Tool 列表，可直接传入 Agent(tools=...)。
    """

    async def _add_node(
        node_type: str,
        info: str,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0,
        description: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        return await graph.add_node(
            node_type=ThinkingNodeType(node_type),
            info=info,
            tags=tags or [],
            confidence=confidence,
            description=description,
            payload=payload or {},
        )

    async def _add_edge(
        edge_type: str,
        source_id: int,
        target_id: int,
        strength: float = 1.0,
        description: str = "",
    ) -> int:
        return await graph.add_edge(
            edge_type=ThinkingEdgeType(edge_type),
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            description=description,
        )

    async def _validate_context(
        node_id: int,
        max_hops: int = 1,
    ) -> str:
        await graph.validate_incremental_context(node_id, max_hops)
        return f"Local context around node {node_id} is valid."

    async def _get_node_info(node_id: int) -> str:
        if node_id not in graph.node_dict:
            return f"Node {node_id} not found."
        node = graph.node_dict[node_id]
        return (
            f"Node {node.id}: type={node.node_type.value}, "
            f"info={node.info!r}, confidence={node.confidence}"
        )

    return [
        Tool(
            name="graph_add_node",
            description="Add a node to the thinking graph. Returns the new node ID.",
            parameters={
                "type": "object",
                "properties": {
                    "node_type": {
                        "type": "string",
                        "enum": [t.value for t in ThinkingNodeType],
                        "description": "Node type",
                    },
                    "info": {"type": "string", "description": "Node content"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 1.0,
                    },
                    "description": {"type": "string", "default": ""},
                },
                "required": ["node_type", "info"],
            },
            handler=_add_node,
        ),
        Tool(
            name="graph_add_edge",
            description="Add an edge between two existing nodes. Returns the new edge ID.",
            parameters={
                "type": "object",
                "properties": {
                    "edge_type": {
                        "type": "string",
                        "enum": [t.value for t in ThinkingEdgeType],
                        "description": "Edge type",
                    },
                    "source_id": {"type": "integer", "description": "Source node ID"},
                    "target_id": {"type": "integer", "description": "Target node ID"},
                    "strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 1.0,
                    },
                    "description": {"type": "string", "default": ""},
                },
                "required": ["edge_type", "source_id", "target_id"],
            },
            handler=_add_edge,
        ),
        Tool(
            name="graph_validate_context",
            description="Validate the local context around a node (incremental check).",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer"},
                    "max_hops": {"type": "integer", "minimum": 0, "default": 1},
                },
                "required": ["node_id"],
            },
            handler=_validate_context,
        ),
        Tool(
            name="graph_get_node_info",
            description="Get basic info of a node by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "integer"},
                },
                "required": ["node_id"],
            },
            handler=_get_node_info,
        ),
    ]
