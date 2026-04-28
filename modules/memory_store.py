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


class MemoryStore:
    """关键信息存储库。支持按标签、文本内容检索。"""

    def __init__(self) -> None:
        self.memories: Dict[int, KeyMemory] = {}
        self._tag_index: Dict[str, Set[int]] = {}
        self._next_id = 0

    def add_memory(
        self,
        content: str,
        source_nodes: Optional[Set[int]] = None,
        tags: Optional[Set[str]] = None,
    ) -> int:
        """添加一条关键记忆。

        Args:
            content: 记忆内容（如公式、重要结论、用户偏好等）。
            source_nodes: 该记忆来源于哪些上下文节点。
            tags: 检索标签。

        Returns:
            新记忆的 id。
        """
        mem_id = self._next_id
        self._next_id += 1

        memory = KeyMemory(
            id=mem_id,
            content=content,
            source_nodes=set(source_nodes) if source_nodes else set(),
            tags=set(tags) if tags else set(),
        )
        self.memories[mem_id] = memory

        for tag in memory.tags:
            self._tag_index.setdefault(tag, set()).add(mem_id)

        return mem_id

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
