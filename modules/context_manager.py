from __future__ import annotations

import json
import re
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
        self.current_turn = 0  # 轮次计数器，用于记忆生命周期管理

    @staticmethod
    def _normalize_formula(content: str) -> str:
        """生成公式类记忆的规范化 key，用于去重。"""
        normalized = content.lower()
        # 去掉 LaTeX 命令
        normalized = re.sub(r'\\[a-zA-Z]+', '', normalized)
        # 去掉特殊符号和空格
        normalized = re.sub(r'[\\{}()\[\]$\s|⟨⟩<>⟨⟩]', '', normalized)
        # 统一上标表示
        normalized = re.sub(r'[²³⁴⁵⁶⁷⁸⁹⁰]', '2', normalized)
        # 只保留字母、数字和基本运算符
        normalized = re.sub(r'[^a-z0-9=+\-*/^]', '', normalized)
        return normalized

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
        """让 LLM Agent 决定如何管理上下文与记忆。支持 JSON 解析失败时重试。"""

        candidate_nodes = list(self.graph.nodes.keys())

        planning_prompt = f"""你是一位上下文与记忆管理专家。请基于以下信息，输出精确的 JSON 决策。

## 可用历史节点
{self._format_nodes_for_decision(candidate_nodes)}

## 已有关键记忆（不会被压缩，可直接引用）
{self._format_memories_for_decision()}

## 用户新问题
{user_message}

## 你的任务（严格按以下规则）

1. **selected_node_ids**: 选择需要加载到当前上下文的节点 id（最多 {self.max_context_nodes} 个）。
   - 优先选择与新问题直接相关的节点
   - 必须包含构成完整推理链的节点

2. **nodes_to_pack**: 指定要压缩其祖先链的节点 id。
   - **关键规则**: 系统会从你指定的节点**向前回溯**，保留最近 {self.pack_keep_recent} 轮，把更早的对话压缩成摘要。
   - **因此你应该填写当前对话链的最新节点**（如 selected_node_ids 中 id 最大的那个），而不是根节点。
   - 不要填写没有祖先的根节点，那样不会产生任何效果。
   - 如果历史很短（≤{self.pack_keep_recent + 1} 轮），请留空 []。

3. **memories_to_extract**: 从对话中提取应永久保存的关键信息。
   - **公式/定义/代码** 等精确信息：必须设置 `"pinned": true, "packable": false`，这样它们会被强制注入所有后续对话，且对应的原始上下文不会被压缩。
   - **一般结论/事实**：可以设置 `"pinned": false, "packable": true`。
   - 每条记忆必须有 `"content"`（内容）和 `"tags"`（标签列表）。

4. **memory_queries**: 列出用于检索已有记忆的关键词或标签。

## 输出格式示例（严格 JSON，不要 markdown 代码块，不要额外文字）

{{"reasoning": "用户询问测量行为，与Node 1的叠加态公式直接相关，因此选择Node 0和Node 1。历史较长，用Node 1触发压缩以节省token。公式是精确信息，需要pinned。","selected_node_ids": [0, 1],"nodes_to_pack": [1],"memories_to_extract": [{{"content": "叠加态公式: |ψ⟩ = α|0⟩ + β|1⟩, |α|²+|β|²=1", "tags": ["公式", "量子计算"], "pinned": true, "packable": false}}],"memory_queries": ["叠加态", "测量"]}}
"""

        for attempt in range(2):
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
                if attempt == 0:
                    # 第一次失败，在 prompt 里追加提醒后重试
                    planning_prompt += "\n\n注意：你上一次的输出不是合法 JSON，请确保本次输出是严格的 JSON 对象，不要有任何额外文字。"
                    continue
                # 两次都失败，回退
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
        """根据 Agent 选择的节点和记忆查询，构建喂给 LLM 的线性上下文。

        记忆注入规则：只有 status == 'committed' 且 turn_id < current_turn 的记忆才能被注入。
        本轮提取的 candidate 记忆不可用于本轮回答，避免循环论证。
        """

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

        # ---- 记忆过滤规则 ----
        def _can_inject(mem: Any) -> bool:
            return mem.status == "committed" and mem.turn_id < self.current_turn

        # 1. 强制注入 pinned memories（但仅限已提交的）
        pinned = [m for m in self.memory_store.get_pinned_memories() if _can_inject(m)]
        skipped_pinned = [m for m in self.memory_store.get_pinned_memories() if not _can_inject(m)]

        # 2. 按需检索非 pinned 记忆
        queried_memories: List[Any] = []
        seen_mem_ids: Set[int] = set(m.id for m in pinned)
        for query in memory_queries:
            for mem in self.memory_store.search_by_text(query):
                if mem.id not in seen_mem_ids and _can_inject(mem):
                    seen_mem_ids.add(mem.id)
                    queried_memories.append(mem)

        # 按标签再检索一次（扩展召回）
        for mem in self.memory_store.get_all_memories():
            if mem.id in seen_mem_ids:
                continue
            for query in memory_queries:
                if any(query.lower() in t.lower() for t in mem.tags):
                    if _can_inject(mem):
                        seen_mem_ids.add(mem.id)
                        queried_memories.append(mem)
                    break

        # 3. 打印注入/跳过报告
        print("\n[Injected Memories]")
        if pinned:
            for m in pinned:
                print(f"  M{m.id} committed (turn {m.turn_id}) [pinned]: {m.content[:50]}...")
        if queried_memories:
            for m in queried_memories:
                print(f"  M{m.id} committed (turn {m.turn_id}): {m.content[:50]}...")
        if not pinned and not queried_memories:
            print("  （本轮无注入记忆）")

        skipped_ids = set(m.id for m in skipped_pinned)
        for m in self.memory_store.get_all_memories():
            if m.id not in seen_mem_ids and m.status == "candidate" and m.id not in skipped_ids:
                skipped_ids.add(m.id)
        if skipped_ids:
            print("\n[Skipped Memories]")
            for mid in sorted(skipped_ids):
                m = self.memory_store.get_memory(mid)
                if m:
                    reason = "本轮提取的 candidate，不可用于本轮回答"
                    print(f"  M{m.id} {m.status} (turn {m.turn_id}): {m.content[:40]}... ({reason})")

        # 4. 组装记忆 block
        memory_blocks: List[str] = []
        if pinned:
            memory_blocks.append("[Pinned Memory 永久记忆]")
            for m in pinned:
                memory_blocks.append(f"  - {m.content}")
        if queried_memories:
            memory_blocks.append("[Retrieved Memory 检索记忆]")
            for m in queried_memories:
                memory_blocks.append(f"  - {m.content}")

        if memory_blocks:
            mem_block = (
                "【系统提示：以下是从历史对话中提取的关键记忆，请在下文回答中结合使用】\n"
                + "\n".join(memory_blocks)
                + "\n【记忆结束】"
            )
            result.append(LLMContext(role="system", content=mem_block))

        # 5. 注入对话历史
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
        applied_ops: Dict[str, Any] = {
            "selected_nodes": [],
            "pack_triggers": [],
            "compressed_nodes": [],
            "summary_nodes": [],
            "extracted_memories": [],
            "memory_queries": [],
            "context_node_ids": [],
        }

        if auto_plan and self.graph.nodes:
            self.current_turn += 1
            decision = await self._make_decision(user_message, parent_ids)
            print(f"\n[Agent 决策] {decision.reasoning}")
            applied_ops["selected_nodes"] = decision.selected_node_ids
            applied_ops["memory_queries"] = decision.memory_queries

            # 1. 执行压缩
            for pack_nid in list(decision.nodes_to_pack):
                if pack_nid not in self.graph.nodes:
                    continue

                chain = self.graph.get_ancestor_chain(
                    pack_nid, max_nodes=99, strategy="longest"
                )
                if len(chain) <= self.pack_keep_recent + 1:
                    alt_candidates = [
                        nid for nid in decision.selected_node_ids
                        if nid in self.graph.nodes
                    ]
                    if alt_candidates:
                        alt_nid = max(alt_candidates)
                        alt_chain = self.graph.get_ancestor_chain(
                            alt_nid, max_nodes=99, strategy="longest"
                        )
                        if len(alt_chain) > self.pack_keep_recent + 1:
                            pack_nid = alt_nid
                            chain = alt_chain
                        else:
                            continue
                    else:
                        continue

                summary_id = await self.graph.compress_ancestors(
                    agent=self.fetcher,
                    node_id=pack_nid,
                    max_nodes=self.max_context_nodes + 4,
                    keep_recent=self.pack_keep_recent,
                )
                if summary_id is not None:
                    applied_ops["pack_triggers"].append(pack_nid)
                    applied_ops["summary_nodes"].append(summary_id)
                    compressed = list(self.graph.nodes[summary_id].summarized_ids)
                    applied_ops["compressed_nodes"].extend(compressed)

            # 2. 提取关键记忆（candidate 状态，本轮不可用）
            existing_contents = {m.content.strip() for m in self.memory_store.get_all_memories()}
            for mem_data in decision.memories_to_extract:
                content = mem_data.get("content", "").strip()
                tags = set(mem_data.get("tags", []))
                pinned = bool(mem_data.get("pinned", False))
                packable = bool(mem_data.get("packable", True))
                if not content:
                    continue
                if content in existing_contents:
                    print(f"[记忆提取] 已存在相同记忆，跳过: {content[:40]}...")
                    continue
                existing_contents.add(content)

                canonical_key = None
                if pinned or "公式" in tags:
                    canonical_key = self._normalize_formula(content)
                    existing = self.memory_store._canonical_index.get(canonical_key)
                    if existing is not None:
                        print(f"[记忆提取] 已存在相同公式（canonical），跳过: {content[:40]}...")
                        continue

                mem_id = self.memory_store.add_memory(
                    content=content,
                    source_nodes=set(decision.selected_node_ids),
                    tags=tags,
                    pinned=pinned,
                    packable=packable,
                    canonical_key=canonical_key,
                    turn_id=self.current_turn,
                    status="candidate",
                )
                applied_ops["extracted_memories"].append(mem_id)
                flag = "[Pinned]" if pinned else ""
                print(f"[记忆提取] Memory {mem_id} {flag} (candidate, turn {self.current_turn}): {content[:60]}...")

            # 3. 构建上下文（本轮 candidate 记忆不会被注入）
            context = self._build_context(
                decision.selected_node_ids,
                decision.memory_queries,
            )
            # 记录实际注入上下文的节点
            if context:
                for ctx in context:
                    # 简单估算：找到对应节点（这里仅做演示，实际可用更精确的方式）
                    pass
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

        # ---- Runtime Applied Ops 打印 ----
        print(f"\n[Runtime Applied Ops]")
        print(f"  selected_nodes: {applied_ops['selected_nodes']}")
        print(f"  pack_triggers: {applied_ops['pack_triggers']}")
        print(f"  compressed_nodes: {applied_ops['compressed_nodes']}")
        print(f"  summary_nodes: {applied_ops['summary_nodes']}")
        print(f"  extracted_memories: {applied_ops['extracted_memories']}")
        print(f"  memory_queries: {applied_ops['memory_queries']}")
        if context:
            total_chars = sum(len(c.content) for c in context)
            print(f"  prompt_context_messages: {len(context)} 条, ~{total_chars} 字符")

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

        # 5. 本轮回答完成后，提交本轮提取的 candidate 记忆为 committed
        if auto_plan and self.graph.nodes:
            committed_ids = self.memory_store.commit_candidates(self.current_turn)
            if committed_ids:
                print(f"\n[Post-answer Commit] 本轮 candidate 记忆已提交为 committed: {committed_ids}")

        # 6. 写入 DAG
        node_id = self.graph.add_node(
            LLMContextPair(
                user_role=LLMContext(role="user", content=user_message),
                llm_role=LLMContext(role="assistant", content=assistant_content),
            ),
            parent_ids=parent_ids,
        )
        return node_id
