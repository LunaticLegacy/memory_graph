from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from modules.llm_fetcher import LLMContext, LLMFetcher


@dataclass
class LLMContextPair:
    """一轮对话的上下文对：用户输入 + 模型回复。"""

    user_role: LLMContext
    llm_role: LLMContext


@dataclass
class LLMContextGraphNode:
    """DAG 中的一个上下文节点。"""

    id: int
    context: LLMContextPair
    parent_ids: Set[int] = field(default_factory=set)   # 支持多父节点（DAG）
    child_ids: Set[int] = field(default_factory=set)

    # 压缩 / 归档状态
    is_summary: bool = False                            # 是否为摘要节点
    archived: bool = False                              # 是否已被摘要覆盖，不再在活跃路径上
    summarized_by: Set[int] = field(default_factory=set) # 被哪些摘要节点覆盖了
    summarized_ids: Set[int] = field(default_factory=set) # 本摘要节点覆盖了哪些原始节点

    # 可选：检索、标签
    summary: str = ""
    tags: Set[str] = field(default_factory=set)


class CyclicGraphError(ValueError):
    """当操作会导致图中出现环时抛出。"""


class LLMContextGraph:
    """基于 DAG 的上下文图。

    允许一个节点拥有多个父节点，从而支持多分支对话历史的合并。
    所有可能产生环的操作都会触发 ``CyclicGraphError``。
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, LLMContextGraphNode] = {}
        self.root_ids: Set[int] = set()     # 没有父节点的节点
        self._next_id = 0

    def add_node(
        self,
        context: LLMContextPair,
        parent_ids: Optional[Set[int]] = None,
    ) -> int:
        """加入新节点。

        该操作为事务操作：验证全部通过后一次性写入，不会留下半成品的图状态。
        新节点本身不会导致环（因为新节点没有子节点，且 id 唯一）。

        Args:
            context: 用户-LLM 对话对。
            parent_ids: 前置上下文节点 id 集合。为空时表示这是一个全新的根对话。

        Returns:
            新分配的节点 id。

        Raises:
            KeyError: 当某个 ``parent_id`` 不存在于图中时抛出。
        """
        node_id = self._next_id
        self._next_id += 1

        parent_ids = set(parent_ids) if parent_ids else set()

        for pid in parent_ids:
            if pid not in self.nodes:
                raise KeyError(f"parent_id {pid} does not exist")

        node = LLMContextGraphNode(
            id=node_id,
            context=context,
            parent_ids=parent_ids,
        )
        self.nodes[node_id] = node

        if not parent_ids:
            self.root_ids.add(node_id)
        else:
            for pid in parent_ids:
                self.nodes[pid].child_ids.add(node_id)
            self.root_ids.discard(node_id)

        return node_id

    def _has_path(self, from_id: int, to_id: int, visited: Optional[Set[int]] = None) -> bool:
        """使用 DFS 检查从 ``from_id`` 出发是否能到达 ``to_id``。

        Args:
            from_id: 起始节点 id。
            to_id: 目标节点 id。
            visited: 已访问节点集合（递归用，外部无需传入）。

        Returns:
            是否存在路径。
        """
        if from_id == to_id:
            return True
        if visited is None:
            visited = set()
        visited.add(from_id)
        for child_id in self.nodes[from_id].child_ids:
            if child_id not in visited and self._has_path(child_id, to_id, visited):
                return True
        return False

    def add_edge(self, child_id: int, parent_id: int) -> None:
        """为已有节点新增一条父节点关系（动态扩展 DAG）。

        Args:
            child_id: 子节点 id。
            parent_id: 父节点 id。

        Raises:
            KeyError: 当节点不存在时抛出。
            CyclicGraphError: 当添加该边会导致环时抛出。
        """
        if child_id not in self.nodes:
            raise KeyError(f"child_id {child_id} does not exist")
        if parent_id not in self.nodes:
            raise KeyError(f"parent_id {parent_id} does not exist")
        if parent_id in self.nodes[child_id].parent_ids:
            return

        # 环检测：若 child_id 已经能到达 parent_id，则添加 parent_id -> child_id 会成环
        if self._has_path(child_id, parent_id):
            raise CyclicGraphError(
                f"Adding edge {parent_id} -> {child_id} would create a cycle"
            )

        self.nodes[child_id].parent_ids.add(parent_id)
        self.nodes[parent_id].child_ids.add(child_id)
        self.root_ids.discard(child_id)

    def get_ancestor_chain(
        self,
        node_id: int,
        max_nodes: int = 8,
        strategy: str = "longest",
    ) -> List[int]:
        """返回从某个根节点到目标节点的一条祖先路径。

        由于 DAG 中一个节点可能有多条来自不同根的路径，
        可通过 ``strategy`` 指定选择策略。

        Args:
            node_id: 目标上下文节点 id。
            max_nodes: 最大返回节点数，超出时截断路径开头（保留靠近目标的后缀）。
            strategy: 路径选择策略，支持：
                - ``"longest"``（默认）：取最长路径；
                - ``"shortest"``：取最短路径；
                - ``"first"``：按 parent_ids 遍历顺序，首个找到的路径。

        Returns:
            从根到目标节点的路径 id 列表（含两端）。

        Raises:
            KeyError: 当 ``node_id`` 不存在时抛出。
            ValueError: 当 ``strategy`` 不合法时抛出。
        """
        if node_id not in self.nodes:
            raise KeyError(f"node_id {node_id} does not exist")
        if strategy not in {"longest", "shortest", "first"}:
            raise ValueError(f"Unknown strategy: {strategy}")

        def dfs(nid: int, path_visited: Set[int]) -> List[int]:
            if nid in path_visited:
                return []
            parents = self.nodes[nid].parent_ids
            if not parents:
                return [nid]

            new_visited = path_visited | {nid}
            candidates: List[List[int]] = []
            for pid in parents:
                sub_path = dfs(pid, new_visited)
                if sub_path:
                    candidates.append(sub_path + [nid])

            if not candidates:
                return [nid]

            if strategy == "longest":
                return max(candidates, key=len)
            elif strategy == "shortest":
                return min(candidates, key=len)
            else:  # first
                return candidates[0]

        chain = dfs(node_id, set())
        if len(chain) > max_nodes:
            chain = chain[-max_nodes:]
        return chain

    def get_all_ancestors(self, node_id: int) -> Set[int]:
        """获取目标节点的所有祖先节点（BFS 展开）。

        Args:
            node_id: 目标节点 id。

        Returns:
            所有祖先节点的 id 集合。

        Raises:
            KeyError: 当 ``node_id`` 不存在时抛出。
        """
        if node_id not in self.nodes:
            raise KeyError(f"node_id {node_id} does not exist")

        ancestors: Set[int] = set()
        queue = list(self.nodes[node_id].parent_ids)
        while queue:
            pid = queue.pop(0)
            if pid not in ancestors:
                ancestors.add(pid)
                queue.extend(self.nodes[pid].parent_ids)
        return ancestors

    def build_linear_context(
        self,
        node_id: int,
        max_nodes: int = 8,
    ) -> List[LLMContext]:
        """把图式上下文压平成 LLM 能吃的线性上下文。

        Args:
            node_id: 终点上下文节点。
            max_nodes: 最大回溯深度，默认 8。

        Returns:
            按对话对顺序展开的 ``LLMContext`` 列表。
        """
        chain = self.get_ancestor_chain(node_id, max_nodes)
        result: List[LLMContext] = []
        for nid in chain:
            pair = self.nodes[nid].context
            result.append(pair.user_role)
            result.append(pair.llm_role)
        return result

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        """从 LLM 响应对象中提取文本内容。"""
        if isinstance(response, str):
            return response
        try:
            if hasattr(response, "choices") and response.choices:
                msg = getattr(response.choices[0], "message", None)
                if msg is None and isinstance(response.choices[0], dict):
                    msg = response.choices[0].get("message")
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content is None and isinstance(msg, dict):
                        content = msg.get("content")
                    return content or ""
        except Exception:
            pass
        return str(response)

    async def compress_ancestors(
        self,
        agent: LLMFetcher,
        node_id: int,
        max_nodes: int = 8,
        keep_recent: int = 2,
        summary_system_prompt: Optional[str] = None,
    ) -> Optional[int]:
        """将指定节点较旧的祖先链压缩为一个摘要节点，并插入图中。

        工作流程：
        1. 获取 ``node_id`` 的最长祖先链；
        2. 保留最近 ``keep_recent`` 个节点不压缩；
        3. 将更早的对话传给 LLM 生成摘要；
        4. 创建摘要节点，替换掉被压缩段与保留段之间的连接关系；
        5. 摘要节点的 ``summarized_ids`` 记录被压缩的所有原始节点。

        Args:
            agent: LLM 调用器。
            node_id: 目标节点（通常是当前对话叶节点）。
            max_nodes: 获取祖先链时的最大长度。
            keep_recent: 保留最近的多少轮对话不压缩。
            summary_system_prompt: 生成摘要时使用的系统提示词。

        Returns:
            摘要节点 id；如果链太短无需压缩则返回 ``None``。

        Raises:
            KeyError: 当 ``node_id`` 不存在时抛出。
        """
        if node_id not in self.nodes:
            raise KeyError(f"node_id {node_id} does not exist")

        chain = self.get_ancestor_chain(node_id, max_nodes, strategy="longest")
        if len(chain) <= keep_recent + 1:
            return None

        # 分割：old_ids 压缩，recent_ids 保留
        old_ids = chain[:-keep_recent] if keep_recent > 0 else chain[:]
        recent_ids = chain[-keep_recent:] if keep_recent > 0 else []

        if not old_ids:
            return None

        # 构造待摘要的对话文本
        lines: List[str] = []
        for nid in old_ids:
            pair = self.nodes[nid].context
            lines.append(f"User: {pair.user_role.content}")
            lines.append(f"Assistant: {pair.llm_role.content}")
            lines.append("")

        prompt = (
            "请对以下多轮对话进行高度浓缩的摘要，保留关键事实、结论和上下文信息。\n"
            "摘要需要足够详细，使得只读摘要就能继续后续对话，不要遗漏重要的用户要求或模型回答。\n\n"
            + "\n".join(lines)
        )

        response = await agent.fetch(
            msg=prompt,
            system_prompt=summary_system_prompt
            or "你是一个对话摘要专家。请生成简洁但信息完整的摘要。",
            temperature=0.3,
            max_tokens=2048,
        )
        summary_text = self._extract_message_content(response)

        # 创建摘要节点
        first_old = self.nodes[old_ids[0]]
        summary_node_id = self._next_id
        self._next_id += 1

        summary_node = LLMContextGraphNode(
            id=summary_node_id,
            context=LLMContextPair(
                user_role=LLMContext(role="user", content="[历史对话摘要]"),
                llm_role=LLMContext(role="assistant", content=summary_text),
            ),
            parent_ids=set(first_old.parent_ids),
            is_summary=True,
            summarized_ids=set(old_ids),
        )
        self.nodes[summary_node_id] = summary_node

        if not summary_node.parent_ids:
            self.root_ids.add(summary_node_id)
        else:
            for pid in summary_node.parent_ids:
                self.nodes[pid].child_ids.add(summary_node_id)

        # 重链：将保留段首节点的父节点中的 old_ids[-1] 替换为摘要节点
        if recent_ids:
            first_recent_id = recent_ids[0]
            last_old_id = old_ids[-1]

            if last_old_id in self.nodes[first_recent_id].parent_ids:
                self.nodes[first_recent_id].parent_ids.discard(last_old_id)
                self.nodes[first_recent_id].parent_ids.add(summary_node_id)
                self.nodes[last_old_id].child_ids.discard(first_recent_id)
                self.nodes[summary_node_id].child_ids.add(first_recent_id)

        # 归档：标记被压缩的旧节点
        for old_id in old_ids:
            self.nodes[old_id].archived = True
            self.nodes[old_id].summarized_by.add(summary_node_id)

        return summary_node_id
