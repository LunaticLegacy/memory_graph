"""Swarm-level orchestration: AgentSwarm holds ExecutionGraph + ThinkingGraph + ToolRegistry.

This module elevates ExecutionGraph from a standalone DAG runner into a
first-class swarm container.  An AgentSwarm owns:

* one ExecutionGraph (execution topology)
* one ThinkingGraph (shared cognitive state)
* one ToolRegistry (global tool pool)
* one LLMFetcher (shared LLM backend)

Agents created inside the swarm automatically receive:

* the swarm's global tools
* ThinkingGraph tools (if ``share_thinking_tools=True``)
* optional ExecutionGraph self-modification tools (if ``share_graph_tools=True``)

The design follows the original Angelus idea: *swarm is the top-level
runtime container; execution graph is its control plane; thinking graph is
its shared memory.*
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from ..agent import Agent
from ..llm_fetcher import LLMFetcher
from ..thinking_graph import ThinkingGraph
from ..tool import Tool, ToolRegistry
from .execution_graph import (
    Edge,
    ExecutionGraph,
    ExecutionNode,
    GraphContext,
    InputNode,
    OutputNode,
    RouterNode,
)

# re-export tools factories for convenience
from ..tools.execution_graph_tools import create_execution_graph_tools
from ..tools.thinking_graph_tools import create_thinking_graph_tools


# ---------------------------------------------------------------------------
# Swarm metadata
# ---------------------------------------------------------------------------

@dataclass
class SwarmSpec:
    """Lightweight declaration of a swarm's intent.  Not a full serialised graph."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AgentSwarm
# ---------------------------------------------------------------------------

class AgentSwarm:
    """Top-level container for a multi-agent system.

    Usage (builder pattern):

        swarm = AgentSwarm(fetcher, name="research")
        swarm.add_tool(web_search)
        swarm.add_agent("planner", "你是规划专家...")
        swarm.add_agent("writer",   "你是写作专家...")
        swarm.connect("input", "planner")
        swarm.connect("planner", "writer")
        swarm.connect("writer", "output")
        ctx = await swarm.run("帮我写篇文章", entry_node_id="input")
    """

    def __init__(
        self,
        llm_fetcher: LLMFetcher,
        name: str = "default",
        spec: Optional[SwarmSpec] = None,
        max_concurrency: Optional[int] = None,
    ) -> None:
        self._llm_fetcher = llm_fetcher
        self._spec = spec or SwarmSpec(name=name)
        self._name = self._spec.name

        # Core subsystems
        self.execution_graph = ExecutionGraph(
            llm_fetcher=llm_fetcher,
            max_concurrency=max_concurrency,
        )
        self.thinking_graph = ThinkingGraph()
        self.tool_registry = ToolRegistry()

        # Agent registry for convenient lookup
        self._agents: Dict[str, Agent] = {}

        # Cached tool factories (created on first access)
        self._thinking_tools: Optional[List[Tool]] = None
        self._graph_tools: Optional[List[Tool]] = None

        # Runtime tracing (lightweight)
        self._run_count = 0
        self._last_context: Optional[GraphContext] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> SwarmSpec:
        return self._spec

    @property
    def agents(self) -> Dict[str, Agent]:
        """Read-only view of registered agents."""
        return dict(self._agents)

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI-compatible function schemas for all registered tools."""
        return self.tool_registry.schemas

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def add_tool(self, tool: Tool) -> "AgentSwarm":
        """Register a tool globally (available to all agents + execution graph)."""
        self.tool_registry.register(tool)
        self.execution_graph.register_tool(tool)
        return self

    def add_tools(self, tools: List[Tool]) -> "AgentSwarm":
        for t in tools:
            self.add_tool(t)
        return self

    def remove_tool(self, tool_name: str) -> Tool:
        """Remove a tool from the global pool."""
        t = self.tool_registry.unregister(tool_name)
        # ExecutionGraph does not have unregister, but we keep it in sync
        # by ignoring missing keys silently.
        self.execution_graph._tool_pool.pop(tool_name, None)
        return t

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(
        self,
        node_id: str,
        system_prompt: str,
        *,
        tools: Optional[List[Tool]] = None,
        share_thinking_tools: bool = True,
        share_graph_tools: bool = False,
        extra_tools: Optional[List[Tool]] = None,
    ) -> "AgentSwarm":
        """Create an Agent, register it, and add it to the execution graph.

        Args:
            node_id: Unique ID inside the swarm / execution graph.
            system_prompt: Agent's system prompt.
            tools: Extra tools passed only to this agent.
            share_thinking_tools: Auto-inject ThinkingGraph tools.
            share_graph_tools: Auto-inject ExecutionGraph self-modification tools.
            extra_tools: Alias for ``tools`` (convenience).
        """
        agent_tools: List[Tool] = []

        # 1. Global tools
        agent_tools.extend(self.tool_registry._tools.values())

        # 2. ThinkingGraph tools (shared cognitive layer)
        if share_thinking_tools:
            if self._thinking_tools is None:
                self._thinking_tools = create_thinking_graph_tools(self.thinking_graph)
            agent_tools.extend(self._thinking_tools)

        # 3. ExecutionGraph self-modification tools
        if share_graph_tools:
            if self._graph_tools is None:
                self._graph_tools = create_execution_graph_tools(self.execution_graph)
            agent_tools.extend(self._graph_tools)

        # 4. Per-agent extras
        extras = extra_tools or tools or []
        if extras:
            agent_tools.extend(extras)

        # Deduplicate by name
        seen: Set[str] = set()
        deduped: List[Tool] = []
        for t in agent_tools:
            if t.name not in seen:
                seen.add(t.name)
                deduped.append(t)

        agent = Agent(
            llm_handler=self._llm_fetcher,
            system_prompt=system_prompt,
            tools=deduped if deduped else None,
        )
        self._agents[node_id] = agent
        self.execution_graph.add_agent_node(agent, node_id=node_id)
        return self

    def remove_agent(self, node_id: str) -> None:
        """Remove an agent from both the registry and the execution graph."""
        self._agents.pop(node_id, None)
        self.execution_graph.remove_node(node_id)

    def get_agent(self, node_id: str) -> Agent:
        return self._agents[node_id]

    def update_agent_prompt(self, node_id: str, system_prompt: str) -> "AgentSwarm":
        """Runtime prompt update for an agent."""
        self.execution_graph.update_agent_prompt(node_id, system_prompt)
        return self

    def add_tool_to_agent(self, node_id: str, tool_name: str) -> "AgentSwarm":
        """Add a global tool to a specific agent at runtime."""
        self.execution_graph.add_tool_to_agent(node_id, tool_name)
        return self

    def remove_tool_from_agent(self, node_id: str, tool_name: str) -> "AgentSwarm":
        """Remove a tool from a specific agent at runtime."""
        self.execution_graph.remove_tool_from_agent(node_id, tool_name)
        return self

    # ------------------------------------------------------------------
    # Topology helpers (delegated to ExecutionGraph, fluent API)
    # ------------------------------------------------------------------

    def add_input(self, node_id: str = "input") -> "AgentSwarm":
        self.execution_graph.add_input_node(node_id=node_id)
        return self

    def add_output(
        self,
        node_id: str = "output",
        collector: Optional[Callable[[List[Any]], Any]] = None,
    ) -> "AgentSwarm":
        self.execution_graph.add_output_node(node_id=node_id, collector=collector)
        return self

    def add_router(
        self,
        node_id: str,
        routes: Dict[str, str],
        agent: Optional[Agent] = None,
        default_route: Optional[str] = None,
    ) -> "AgentSwarm":
        self.execution_graph.add_router_node(
            routes=routes,
            agent=agent,
            default_route=default_route,
            node_id=node_id,
        )
        return self

    def add_join(self, node_id: str, strategy: str = "all") -> "AgentSwarm":
        self.execution_graph.add_join_node(strategy=strategy, node_id=node_id)
        return self

    def add_tool_node(self, tool_name: str, node_id: Optional[str] = None) -> "AgentSwarm":
        """Add a global tool as an execution-graph node."""
        tool = self.tool_registry.get(tool_name)
        self.execution_graph.add_tool_node(tool, node_id=node_id)
        return self

    def connect(
        self,
        source_id: str,
        target_id: str,
        label: Optional[str] = None,
    ) -> "AgentSwarm":
        self.execution_graph.connect(source_id, target_id, label)
        return self

    def disconnect(
        self,
        source_id: str,
        target_id: str,
        label: Optional[str] = None,
    ) -> "AgentSwarm":
        self.execution_graph.disconnect(source_id, target_id, label)
        return self

    def set_timeout(self, node_id: str, seconds: float) -> "AgentSwarm":
        self.execution_graph.set_node_timeout(node_id, seconds)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        initial_input: Any = None,
        entry_node_id: Optional[str] = None,
    ) -> GraphContext:
        """Run the swarm's execution graph.

        The ThinkingGraph is shared across all agents, so any agent that
        calls ``thinking_graph_add_node`` during the run will mutate the
        swarm-level cognitive state.
        """
        self._run_count += 1
        ctx = await self.execution_graph.run(
            initial_input=initial_input,
            entry_node_id=entry_node_id,
        )
        self._last_context = ctx
        return ctx

    @property
    def last_context(self) -> Optional[GraphContext]:
        """The result of the most recent ``run()``."""
        return self._last_context

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot of the swarm (not fully round-trippable because Agents
        carry live callables, but good for debugging / UI display)."""
        return {
            "spec": {
                "name": self._spec.name,
                "description": self._spec.description,
                "version": self._spec.version,
                "metadata": dict(self._spec.metadata),
            },
            "execution_graph": self.execution_graph.to_dict(),
            "thinking_graph": self.thinking_graph.to_dict(),
            "agents": {
                nid: {
                    "system_prompt": a.system_prompt[:200],
                    "tool_count": len(a.tool_registry._tools),
                }
                for nid, a in self._agents.items()
            },
            "tool_names": sorted(self.tool_registry._tools.keys()),
            "run_count": self._run_count,
        }

    def __repr__(self) -> str:
        return (
            f"AgentSwarm({self._name!r}, "
            f"agents={len(self._agents)}, "
            f"tools={len(self.tool_registry._tools)}, "
            f"runs={self._run_count})"
        )
