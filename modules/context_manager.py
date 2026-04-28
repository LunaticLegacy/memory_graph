from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from modules.context_graph import LLMContextGraph, LLMContextGraphNode, LLMContextPair
from modules.llm_fetcher import LLMContext, LLMFetcher
from modules.memory_store import MemoryStore


@dataclass
class ContextDecision:
    """Agent 对上下文管理的决策结果。"""

    reasoning: str
    selected_node_ids: List[int] = field(default_factory=list)
    nodes_to_pack: List[int] = field(default_factory=list)
    memories_to_extract: List[Dict[str, Any]] = field(default_factory=list)
    memory_queries: List[str] = field(default_factory=list)


class ContextManager:
    """整合上下文图、记忆存储与 LLM 决策的上下文管理器。

    核心能力：
    1. **上下文选择**：Agent 根据当前问题，自主选择需要加载哪些历史节点；
    2. **上下文压缩**：Agent 决定哪些旧节点应该被打包成摘要；
    3. **关键信息提取**：Agent 从对话中提取重要公式/结论/事实，存入 MemoryStore，
       这些内容独立于图，后续可直接引用，不会被压缩覆盖。
    """

    def __init__(
        self,
        fetcher: LLMFetcher,
        max_context_nodes: int = 6,
        pack_keep_recent: int = 1,
    ) -> None:
        self.fetcher = fetcher
        self.graph = LLMContextGraph()
        self.memory_store = MemoryStore()
        self.max_context_nodes = max_context_nodes
        self.pack_keep_recent = pack_keep_recent

    # ------------------------------------------------------------------
    # 内部工具：格式化信息供决策使用
    # ------------------------------------------------------------------

    def _format_nodes_for_decision(self, candidate_nodes: List[int]) -> str:
        lines: List[str] = []
        for nid in candidate_nodes:
            node = self.graph.nodes[nid]
            pair = node.context
            u = pair.user_role.content.replace("\n", " ")[:80]
            a = pair.llm_role.content.replace("\n", " ")[:80]
            lines.append(f"Node {nid}: parents={node.parent_ids} children={node.child_ids}")
            lines.append(f'  User: "{u}..."')
            lines.append(f'  Assistant: "{a}..."')
            if node.summarized_ids:
                lines.append(f"  [summary node, compressed from {node.summarized_ids}]")
            lines.append("")
        return "\n".join(lines)

    def _format_memories_for_decision(self) -> str:
        memories = self.memory_store.get_all_memories()
        if not memories:
            return "（暂无已提取的关键记忆）"
        lines: List[str] = []
        for mem in memories:
            content = mem.content.replace("\n", " ")[:100]
            lines.append(f"Memory {mem.id} [tags={mem.tags}]: {content}...")
        return "\n".join(lines)

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        """从 LLM 输出中安全提取 JSON。"""
        raw = raw.strip()
        # 去掉可能的 markdown 代码块
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        return json.loads(raw)

    # ------------------------------------------------------------------
    # 核心：Agent 决策
    # ------------------------------------------------------------------

    async def _make_decision(
        self,
        user_message: str,
        parent_ids: Optional[Set[int]],
    ) -> ContextDecision:
        """让 LLM Agent 决定如何管理上下文与记忆。"""

        candidate_nodes = list(self.graph.nodes.keys())

        planning_prompt = f"""你是一位上下文与记忆管理专家。请基于以下信息，输出精确的 JSON 决策。

## 可用历史节点
{self._format_nodes_for_decision(candidate_nodes)}

## 已有关键记忆（不会被压缩，可直接引用）
{self._format_memories_for_decision()}

## 用户新问题
{user_message}

## 你的任务
1. **selected_node_ids**: 选择需要加载到当前上下文的节点 id（最多 {self.max_context_nodes} 个）。优先选择与新问题直接相关的节点，以及构成完整推理链的节点。
2. **nodes_to_pack**: 选择应该被压缩打包的旧节点 id（通常是非常久远、细节已不重要的节点）。压缩后这些节点的内容会变成摘要，节省 token。
3. **memories_to_extract**: 从对话中提取应永久保存的关键信息（如重要公式、核心结论、用户明确要求记住的偏好或事实）。这些会被存入独立记忆库，永远不会被压缩。
4. **memory_queries**: 列出用于检索已有记忆的关键词或标签（如 "formula", "量子计算"）。

## 输出格式（严格 JSON，不要 markdown 代码块）
{{"reasoning": "你的思考过程（中文）","selected_node_ids": [1, 3],"nodes_to_pack": [0],"memories_to_extract": [{{"content": "关键信息内容", "tags": ["tag1"]}}],"memory_queries": ["关键词1"]}}
"""

        try:
            response = await self.fetcher.fetch(
                msg=planning_prompt,
                system_prompt="你只输出合法 JSON，不要添加任何 markdown 标记、解释文字或换行符包裹。",
                temperature=0.1,
                max_tokens=2048,
            )
            raw = self.graph._extract_message_content(response)
            data = self._extract_json(raw)

            return ContextDecision(
                reasoning=data.get("reasoning", ""),
                selected_node_ids=data.get("selected_node_ids", []),
                nodes_to_pack=data.get("nodes_to_pack", []),
                memories_to_extract=data.get("memories_to_extract", []),
                memory_queries=data.get("memory_queries", []),
            )
        except Exception as exc:
            # 决策失败时回退到安全默认值
            fallback_nodes = list(parent_ids or [])
            return ContextDecision(
                reasoning=f"决策解析失败（{exc}），回退到默认最长链",
                selected_node_ids=fallback_nodes,
                nodes_to_pack=[],
                memories_to_extract=[],
                memory_queries=[],
            )

    # ------------------------------------------------------------------
    # 核心：构建最终上下文
    # ------------------------------------------------------------------

    def _build_context(
        self,
        selected_node_ids: List[int],
        memory_queries: List[str],
    ) -> List[LLMContext]:
        """根据 Agent 选择的节点和记忆查询，构建喂给 LLM 的线性上下文。"""

        # 收集选中节点的祖先链（去重）
        all_nids: Set[int] = set()
        for nid in selected_node_ids:
            if nid in self.graph.nodes:
                chain = self.graph.get_ancestor_chain(
                    nid, max_nodes=self.max_context_nodes, strategy="longest"
                )
                all_nids.update(chain)
                all_nids.add(nid)

        sorted_nids = sorted(all_nids)
        result: List[LLMContext] = []

        # 注入相关记忆（system 消息形式）
        relevant_memories: List[Any] = []
        seen_mem_ids: Set[int] = set()
        for query in memory_queries:
            for mem in self.memory_store.search_by_text(query):
                if mem.id not in seen_mem_ids:
                    seen_mem_ids.add(mem.id)
                    relevant_memories.append(mem)

        # 按标签再检索一次（扩展召回）
        for mem in self.memory_store.get_all_memories():
            if mem.id in seen_mem_ids:
                continue
            for query in memory_queries:
                if any(query.lower() in t.lower() for t in mem.tags):
                    seen_mem_ids.add(mem.id)
                    relevant_memories.append(mem)
                    break

        if relevant_memories:
            mem_lines = [f"- {m.content}" for m in relevant_memories]
            mem_block = (
                "【已提取的关键记忆（永久保留，不会被压缩）】\n"
                + "\n".join(mem_lines)
                + "\n【记忆结束】"
            )
            result.append(LLMContext(role="system", content=mem_block))

        # 注入对话历史
        for nid in sorted_nids:
            pair = self.graph.nodes[nid].context
            result.append(pair.user_role)
            result.append(pair.llm_role)

        return result

    # ------------------------------------------------------------------
    # 公开接口：带规划的对话
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        parent_ids: Optional[Set[int]] = None,
        auto_plan: bool = True,
    ) -> int:
        """执行一轮对话。

        当 ``auto_plan=True`` 且图中已有节点时，会先让 Agent 做上下文管理决策，
        再调用 LLM 生成回复。

        Args:
            user_message: 用户输入。
            parent_ids: 显式指定的父节点（用于图的拓扑连接）。
            auto_plan: 是否启用 Agent 自主决策。

        Returns:
            新创建的节点 id。
        """
        context: Optional[List[LLMContext]] = None

        if auto_plan and self.graph.nodes:
            decision = await self._make_decision(user_message, parent_ids)
            print(f"\n[Agent 决策] {decision.reasoning}")

            # 1. 执行压缩
            for pack_nid in decision.nodes_to_pack:
                if pack_nid not in self.graph.nodes:
                    continue
                summary_id = await self.graph.compress_ancestors(
                    agent=self.fetcher,
                    node_id=pack_nid,
                    max_nodes=self.max_context_nodes + 2,
                    keep_recent=self.pack_keep_recent,
                )
                if summary_id is not None:
                    print(f"[压缩] Node {pack_nid} 的祖先已压缩为 Node {summary_id}")

            # 2. 提取关键记忆
            for mem_data in decision.memories_to_extract:
                content = mem_data.get("content", "").strip()
                tags = set(mem_data.get("tags", []))
                if content:
                    mem_id = self.memory_store.add_memory(
                        content=content,
                        source_nodes=set(decision.selected_node_ids),
                        tags=tags,
                    )
                    print(f"[记忆提取] Memory {mem_id}: {content[:60]}...")

            # 3. 构建上下文
            context = self._build_context(
                decision.selected_node_ids,
                decision.memory_queries,
            )
        else:
            # 简单模式：取最长链
            if parent_ids:
                all_nids: Set[int] = set()
                for pid in parent_ids:
                    all_nids.update(
                        self.graph.get_ancestor_chain(pid, max_nodes=self.max_context_nodes)
                    )
                    all_nids.add(pid)
                context = []
                for nid in sorted(all_nids):
                    pair = self.graph.nodes[nid].context
                    context.append(pair.user_role)
                    context.append(pair.llm_role)

        # 4. 调用 LLM
        print(f"\n[User] {user_message}")
        print("[Assistant] ", end="", flush=True)

        assistant_content = ""
        async for chunk in self.fetcher.fetch_stream(
            msg=user_message,
            system_prompt="你是一个知识渊博、表达简洁的 AI 助手。",
            temperature=0.7,
            max_tokens=1024,
            output_reasoning=False,
            prev_messages=context,
        ):
            print(chunk, end="", flush=True)
            assistant_content += chunk
        print("")

        # 5. 写入 DAG
        node_id = self.graph.add_node(
            LLMContextPair(
                user_role=LLMContext(role="user", content=user_message),
                llm_role=LLMContext(role="assistant", content=assistant_content),
            ),
            parent_ids=parent_ids,
        )
        return node_id
