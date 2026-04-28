from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class KeyMemory:
    """从对话中提取的关键信息（记忆）。

    被记忆的内容独立于上下文图，不会因压缩/打包而丢失。
    """

    id: int
    content: str
    source_nodes: Set[int] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    pinned: bool = False       # 为 True 时，该记忆会被强制注入所有后续对话的 system prompt
    packable: bool = True      # 为 False 时，该记忆对应的原始上下文不应被压缩
    status: str = "candidate"  # "committed" | "candidate" | "rejected"
    turn_id: int = 0           # 创建时的轮次编号


class MemoryStore:
    """关键信息存储库。支持按标签、文本内容检索，支持 canonical key 去重。"""

    def __init__(self) -> None:
        self.memories: Dict[int, KeyMemory] = {}
        self._tag_index: Dict[str, Set[int]] = {}
        self._canonical_index: Dict[str, int] = {}   # canonical_key -> mem_id
        self._next_id = 0

    def add_memory(
        self,
        content: str,
        source_nodes: Optional[Set[int]] = None,
        tags: Optional[Set[str]] = None,
        pinned: bool = False,
        packable: bool = True,
        canonical_key: Optional[str] = None,
        turn_id: int = 0,
        status: str = "candidate",
    ) -> int:
        """添加一条关键记忆。

        Args:
            content: 记忆内容。
            source_nodes: 来源上下文节点。
            tags: 检索标签。
            pinned: 是否强制注入所有后续对话。
            packable: 对应的原始上下文是否允许被压缩。
            canonical_key: 规范化 key，用于去重。
            turn_id: 创建轮次。
            status: 初始状态（默认 candidate，回答后由 runtime 提升为 committed）。

        Returns:
            记忆 id（新创建或已存在）。
        """
        if canonical_key and canonical_key in self._canonical_index:
            existing_id = self._canonical_index[canonical_key]
            self.memories[existing_id].source_nodes.update(source_nodes or set())
            return existing_id

        mem_id = self._next_id
        self._next_id += 1

        memory = KeyMemory(
            id=mem_id,
            content=content,
            source_nodes=set(source_nodes) if source_nodes else set(),
            tags=set(tags) if tags else set(),
            pinned=pinned,
            packable=packable,
            status=status,
            turn_id=turn_id,
        )
        self.memories[mem_id] = memory

        for tag in memory.tags:
            self._tag_index.setdefault(tag, set()).add(mem_id)

        if canonical_key:
            self._canonical_index[canonical_key] = mem_id

        return mem_id

    def get_pinned_memories(self) -> List[KeyMemory]:
        """获取所有被标记为 pinned 的记忆。"""
        return [m for m in self.memories.values() if m.pinned]

    def get_committed_memories(self) -> List[KeyMemory]:
        """获取所有已提交（committed）的记忆。"""
        return [m for m in self.memories.values() if m.status == "committed"]

    def commit_candidates(self, turn_id: int) -> List[int]:
        """将指定轮次创建的 candidate 记忆提升为 committed。

        Returns:
            被提升的记忆 id 列表。
        """
        committed_ids: List[int] = []
        for mem in self.memories.values():
            if mem.status == "candidate" and mem.turn_id == turn_id:
                mem.status = "committed"
                committed_ids.append(mem.id)
        return committed_ids

    def get_memory(self, mem_id: int) -> Optional[KeyMemory]:
        """按 id 获取记忆。"""
        return self.memories.get(mem_id)

    def search_by_tags(self, tags: Set[str]) -> List[KeyMemory]:
        """按标签交集检索记忆。"""
        if not tags:
            return []
        candidate_ids: Optional[Set[int]] = None
        for tag in tags:
            ids = self._tag_index.get(tag, set())
            if candidate_ids is None:
                candidate_ids = set(ids)
            else:
                candidate_ids &= ids
            if not candidate_ids:
                return []
        return [self.memories[mid] for mid in sorted(candidate_ids or [])]

    def search_by_text(self, query: str) -> List[KeyMemory]:
        """按文本模糊匹配检索记忆（包含标签内容）。"""
        query_lower = query.lower()
        results: List[KeyMemory] = []
        for mem in self.memories.values():
            if query_lower in mem.content.lower():
                results.append(mem)
                continue
            if any(query_lower in t.lower() for t in mem.tags):
                results.append(mem)
                continue
        return results

    def get_all_memories(self) -> List[KeyMemory]:
        """返回所有记忆，按 id 排序。"""
        return [self.memories[mid] for mid in sorted(self.memories.keys())]

    def remove_memory(self, mem_id: int) -> bool:
        """删除指定记忆。"""
        mem = self.memories.pop(mem_id, None)
        if mem is None:
            return False
        for tag in mem.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(mem_id)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        return True
