from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from ..agent import Agent
from ..tool import Tool


# ---------------------------------------------------------------------------
# 边定义
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """执行图中的一条有向边。"""

    source_id: str      # 源
    target_id: str      # 目标
    label: Optional[str] = None  # 路由标签，用于条件分支


# ---------------------------------------------------------------------------
# 执行上下文
# ---------------------------------------------------------------------------

class GraphContext:
    """执行图运行时的上下文，保存节点间的数据流和状态。"""

    def __init__(self, graph: ExecutionGraph):
        self.graph = graph
        self.node_inputs: Dict[str, List[Any]] = {}
        self.node_outputs: Dict[str, Any] = {}
        self.executed: Set[str] = set()
        self.metadata: Dict[str, Any] = {}

    def get_output(self, node_id: str) -> Any:
        return self.node_outputs.get(node_id)

    def get_inputs(self, node_id: str) -> List[Any]:
        return self.node_inputs.get(node_id, [])


# ---------------------------------------------------------------------------
# 节点基类与实现
# ---------------------------------------------------------------------------

class ExecutionNode(ABC):
    """执行图节点的抽象基类。"""

    def __init__(self, node_id: str, node_type: str):
        self.node_id = node_id
        self.node_type = node_type

    @abstractmethod
    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Any:
        """执行节点逻辑，返回结果。"""
        ...


class AgentNode(ExecutionNode):
    """包装 Agent 的执行节点。"""

    def __init__(self, node_id: str, agent: Agent):
        super().__init__(node_id, "agent")
        self.agent = agent

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> str:
        if not inputs:
            msg = "请开始执行任务。"
        elif len(inputs) == 1:
            msg = str(inputs[0])
        else:
            parts = [f"[输入 {i + 1}]\n{str(inp)}" for i, inp in enumerate(inputs)]
            msg = "\n\n".join(parts)
        return await self.agent.round_call(msg, stream=False, max_turns=3)


class ToolNode(ExecutionNode):
    """包装 Tool 的执行节点。"""

    def __init__(self, node_id: str, tool: Tool):
        super().__init__(node_id, "tool")
        self.tool = tool

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Any:
        if len(inputs) == 1 and isinstance(inputs[0], dict):
            args = inputs[0]
        elif len(inputs) == 1:
            args = {"input": inputs[0]}
        else:
            args = {"inputs": inputs}
        try:
            return await self.tool.execute(**args)
        except Exception as exc:
            return {"error": str(exc), "tool": self.tool.name}


class RouterNode(ExecutionNode):
    """路由节点：根据输入决定路由标签，实现条件分支。"""

    def __init__(
        self,
        node_id: str,
        routes: Dict[str, str],
        agent: Optional[Agent] = None,
        default_route: Optional[str] = None,
    ):
        super().__init__(node_id, "router")
        self.routes = routes
        self.agent = agent
        self.default_route = default_route or (list(routes.keys())[0] if routes else None)

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Dict[str, Any]:
        content = "\n\n".join(str(i) for i in inputs)

        if self.agent and len(self.routes) > 1:
            routes_desc = "\n".join(f"- {k}: {v}" for k, v in self.routes.items())
            prompt = (
                f"请根据以下输入内容，选择最合适的路由方向。\n\n"
                f"可选方向：\n{routes_desc}\n\n"
                f"输入内容：\n{content}\n\n"
                f"请只输出一个路由标签（{list(self.routes.keys())}），不要输出其他内容。"
            )
            result = await self.agent.round_call(prompt, stream=False, max_turns=1)
            selected = self.default_route
            for label in self.routes:
                if label in result:
                    selected = label
                    break
            return {"route": selected, "raw": result, "input": content}

        return {"route": self.default_route, "input": content}


class InputNode(ExecutionNode):
    """入口节点，直接透传输入。"""

    def __init__(self, node_id: str):
        super().__init__(node_id, "input")

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Any:
        return inputs[0] if inputs else None


class OutputNode(ExecutionNode):
    """出口节点，收集并返回结果。"""

    def __init__(self, node_id: str, collector: Optional[Callable] = None):
        super().__init__(node_id, "output")
        self.collector = collector or (lambda x: x)

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Any:
        return self.collector(inputs)


class JoinNode(ExecutionNode):
    """汇聚节点：将多个上游输入聚合成结构化输出。

    在当前 DAG 调度下，JoinNode 被调用时所有上游天然已执行完毕。
    它的价值在于语义明确和输出格式化。
    """

    def __init__(self, node_id: str, strategy: str = "all"):
        super().__init__(node_id, "join")
        self.strategy = strategy  # "all", "first"

    async def run(self, ctx: GraphContext, inputs: List[Any]) -> Dict[str, Any]:
        if self.strategy == "first":
            return {"result": inputs[0] if inputs else None, "inputs": inputs}
        return {"results": inputs, "count": len(inputs)}


# ---------------------------------------------------------------------------
# 执行图本体
# ---------------------------------------------------------------------------

class ExecutionGraph:
    """Agent Swarm 的执行图，支持运行时动态增删节点、工具与修改提示词。

    调度策略（事件驱动）：
    - 节点一旦就绪（所有上游已执行完毕）立即启动，不需要等待整层完成。
    - 并发控制：可通过 max_concurrency 限制同时运行的节点数。
    - 超时控制：可为单个节点设置 timeout。
    - 并行：DAG 天然支持并行分支，多个无依赖的节点会同时启动。
    """

    def __init__(
        self,
        llm_fetcher: Optional[Any] = None,
        max_concurrency: Optional[int] = None,
    ):
        """

        Args:
            max_concurrency: 最大并发数，即最大可同时运行节点数。
        """
        self._nodes: Dict[str, ExecutionNode] = {}
        self._edges: List[Edge] = []
        self._lock = asyncio.Lock()
        self._node_counter = 0

        # 用于运行时动态创建 Agent
        self._llm_fetcher = llm_fetcher

        # 全局工具池
        self._tool_pool: Dict[str, Tool] = {}

        # 并发控制
        self._semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        # 节点超时配置
        self._node_timeouts: Dict[str, float] = {}

    # --- 内部辅助 ---

    def _alloc_id(self, prefix: str = "node") -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def _upstream_of(self, node_id: str) -> Set[str]:
        return {e.source_id for e in self._edges if e.target_id == node_id}

    def _downstream_of(self, node_id: str) -> List[Edge]:
        return [e for e in self._edges if e.source_id == node_id]

    def _find_entry_nodes(self) -> List[str]:
        """没有上游的节点被视为入口。"""
        return [nid for nid in self._nodes if not self._upstream_of(nid)]

    # --- 工具池 ---

    def register_tool(self, tool: Tool) -> None:
        """向全局工具池注册一个工具。"""
        self._tool_pool[tool.name] = tool

    def unregister_tool(self, tool_name: str) -> Tool:
        """从全局工具池移除一个工具。"""
        if tool_name not in self._tool_pool:
            raise KeyError(f"Tool '{tool_name}' not in pool")
        return self._tool_pool.pop(tool_name)

    def get_tool(self, tool_name: str) -> Tool:
        return self._tool_pool[tool_name]

    @property
    def tool_pool(self) -> Dict[str, Tool]:
        return dict(self._tool_pool)

    # --- 节点生命周期（动态增删） ---

    def add_agent_node(
        self,
        agent: Agent,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or self._alloc_id("agent")
        self._nodes[nid] = AgentNode(nid, agent)
        return nid

    def add_tool_node(
        self,
        tool: Tool,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or self._alloc_id("tool")
        self._nodes[nid] = ToolNode(nid, tool)
        return nid

    def add_router_node(
        self,
        routes: Dict[str, str],
        agent: Optional[Agent] = None,
        default_route: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or self._alloc_id("router")
        self._nodes[nid] = RouterNode(nid, routes, agent, default_route)
        return nid

    def add_input_node(self, node_id: Optional[str] = None) -> str:
        nid = node_id or self._alloc_id("input")
        self._nodes[nid] = InputNode(nid)
        return nid

    def add_output_node(
        self,
        collector: Optional[Callable] = None,
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or self._alloc_id("output")
        self._nodes[nid] = OutputNode(nid, collector)
        return nid

    def add_join_node(
        self,
        strategy: str = "all",
        node_id: Optional[str] = None,
    ) -> str:
        nid = node_id or self._alloc_id("join")
        self._nodes[nid] = JoinNode(nid, strategy)
        return nid

    def remove_node(self, node_id: str) -> None:
        """删除节点并清理相关边。"""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not found")
        del self._nodes[node_id]
        self._edges = [
            e
            for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        self._node_timeouts.pop(node_id, None)

    def get_node(self, node_id: str) -> ExecutionNode:
        return self._nodes[node_id]

    @property
    def nodes(self) -> Dict[str, ExecutionNode]:
        return dict(self._nodes)

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    # --- 边管理 ---

    def connect(
        self,
        source_id: str,
        target_id: str,
        label: Optional[str] = None,
    ) -> None:
        if source_id not in self._nodes:
            raise KeyError(f"Source node {source_id} not found")
        if target_id not in self._nodes:
            raise KeyError(f"Target node {target_id} not found")
        self._edges.append(Edge(source_id, target_id, label))

    def disconnect(
        self,
        source_id: str,
        target_id: str,
        label: Optional[str] = None,
    ) -> None:
        self._edges = [
            e
            for e in self._edges
            if not (
                e.source_id == source_id
                and e.target_id == target_id
                and (label is None or e.label == label)
            )
        ]

    # --- 动态修改 Agent 配置 ---

    def update_agent_prompt(self, node_id: str, system_prompt: str) -> None:
        """运行时修改指定 Agent 节点的系统提示词。"""
        node = self._nodes.get(node_id)
        if not isinstance(node, AgentNode):
            raise TypeError(f"Node {node_id} is not an agent node")
        node.agent.update_system_prompt(system_prompt)

    def add_tool_to_agent(self, node_id: str, tool_name: str) -> None:
        """运行时给指定 Agent 节点增加一个来自全局工具池的工具。"""
        node = self._nodes.get(node_id)
        if not isinstance(node, AgentNode):
            raise TypeError(f"Node {node_id} is not an agent node")
        tool = self._tool_pool.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found in global pool")
        node.agent.add_tool(tool)

    def remove_tool_from_agent(self, node_id: str, tool_name: str) -> None:
        """运行时从指定 Agent 节点移除一个工具。"""
        node = self._nodes.get(node_id)
        if not isinstance(node, AgentNode):
            raise TypeError(f"Node {node_id} is not an agent node")
        node.agent.remove_tool(tool_name)

    def set_node_timeout(self, node_id: str, timeout: float) -> None:
        """为指定节点设置执行超时（秒）。"""
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} not found")
        self._node_timeouts[node_id] = timeout

    # --- 执行：事件驱动调度 ---

    async def run(
        self,
        initial_input: Any = None,
        entry_node_id: Optional[str] = None,
    ) -> GraphContext:
        """启动执行图，返回执行后的上下文。

        调度逻辑：
        1. 找出所有就绪节点（上游已全部执行完毕）并启动。
        2. 节点完成后立即将输出路由给下游。
        3. 再次检查是否有新节点就绪，周而复始直到没有运行中的节点。
        """
        ctx = GraphContext(self)

        entry = entry_node_id
        if entry is None:
            entries = self._find_entry_nodes()
            if not entries:
                raise ValueError("No entry node found in graph")
            entry = entries[0]

        if initial_input is not None:
            ctx.node_inputs[entry] = [initial_input]

        completed_queue: asyncio.Queue[str] = asyncio.Queue()
        running: Set[str] = set()

        async def worker(nid: str):
            """执行单个节点，完成后放入完成队列。"""
            try:
                sem = self._semaphore
                if sem:
                    async with sem:
                        result = await self._run_node_with_timeout(nid, ctx)
                else:
                    result = await self._run_node_with_timeout(nid, ctx)
            except asyncio.TimeoutError:
                result = {
                    "error": "Node execution timed out",
                    "node_id": nid,
                    "node_type": self._nodes[nid].node_type,
                }
            except Exception as exc:
                result = {
                    "error": str(exc),
                    "node_id": nid,
                    "node_type": self._nodes[nid].node_type,
                }

            # 保存结果并路由到下游
            ctx.node_outputs[nid] = result
            ctx.executed.add(nid)
            running.discard(nid)

            for edge in self._downstream_of(nid):
                if edge.label is not None:
                    route = self._extract_route(result)
                    if route != edge.label:
                        continue
                ctx.node_inputs.setdefault(edge.target_id, []).append(result)

            await completed_queue.put(nid)

        def try_start(nid: str) -> bool:
            """尝试启动一个节点。返回是否成功启动。"""
            if nid in ctx.executed or nid in running:
                return False
            upstream = self._upstream_of(nid)
            if upstream and not all(u in ctx.executed for u in upstream):
                return False
            if not upstream and nid not in ctx.node_inputs:
                return False
            running.add(nid)
            asyncio.create_task(worker(nid))
            return True

        # 启动初始就绪节点
        for nid in list(self._nodes.keys()):
            try_start(nid)

        # 事件驱动主循环：节点完成 → 尝试启动新就绪节点
        while running:
            completed_nid = await completed_queue.get()
            for candidate in list(self._nodes.keys()):
                try_start(candidate)

        return ctx

    async def _run_node_with_timeout(self, nid: str, ctx: GraphContext) -> Any:
        node = self._nodes[nid]
        inputs = ctx.node_inputs.get(nid, [])
        timeout = self._node_timeouts.get(nid)

        coro = node.run(ctx, inputs)
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    def _extract_route(self, result: Any) -> Optional[str]:
        if isinstance(result, dict):
            return result.get("route")
        return None

    # --- 序列化 ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {
                nid: {"type": n.node_type, "id": nid}
                for nid, n in self._nodes.items()
            },
            "edges": [
                {"source": e.source_id, "target": e.target_id, "label": e.label}
                for e in self._edges
            ],
        }
