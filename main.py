import asyncio
import os
from modules import LLMFetcher, LLMContextGraph, ContextManager


def print_final_state(cm: ContextManager, latest_node_id: int) -> None:
    print(f"\n{'='*50}")
    print("最终状态总结")
    print("="*50)

    # ---- Active Graph ----
    print("\n【Active Graph 活跃图】")
    active_nodes = {nid for nid, n in cm.graph.nodes.items() if not n.archived}
    for nid, node in sorted(cm.graph.nodes.items()):
        if node.archived:
            continue
        marker = ""
        if node.is_summary:
            marker += " [摘要节点]"
        if nid == latest_node_id:
            marker += " <-- 最新节点"
        print(f"  Node {nid}: parents={node.parent_ids}, children={node.child_ids}{marker}")

    # ---- Archived Graph ----
    archived = {nid for nid, n in cm.graph.nodes.items() if n.archived}
    if archived:
        print("\n【Archived Graph 已归档节点】")
        for nid in sorted(archived):
            node = cm.graph.nodes[nid]
            print(f"  Node {nid}: summarized_by={node.summarized_by}, children={node.child_ids}")

    # ---- Active Path ----
    active_chain = cm.graph.get_ancestor_chain(latest_node_id, max_nodes=8, strategy="longest")
    print(f"\n活跃上下文路径 (Node {latest_node_id}): {active_chain}")
    print(f"所有祖先: {cm.graph.get_all_ancestors(latest_node_id)}")

    # ---- Pinned Memory ----
    print("\n【Pinned Memory 永久记忆】")
    pinned = cm.memory_store.get_pinned_memories()
    if not pinned:
        print("  （暂无）")
    else:
        for mem in pinned:
            print(f"  Memory {mem.id} [tags={mem.tags}]: {mem.content}")

    # ---- All Memory ----
    print("\n【All Memory 全部记忆】")
    all_mem = cm.memory_store.get_all_memories()
    if not all_mem:
        print("  （暂无）")
    else:
        for mem in all_mem:
            flags = []
            if mem.pinned:
                flags.append("pinned")
            if not mem.packable:
                flags.append("!packable")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"  Memory {mem.id}{flag_str} [tags={mem.tags}]: {mem.content[:80]}...")


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable DEEPSEEK_API_KEY is not set. "
            "Run: source ./load_sk.sh"
        )

    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    cm = ContextManager(
        fetcher=fetcher,
        max_context_nodes=6,
        pack_keep_recent=1,
    )

    # ============================================================
    # Round 0: 基础介绍
    # ============================================================
    n0 = await cm.chat(
        user_message="请用一段话介绍日本御宅族（Otaku）文化的起源。",
        auto_plan=False,
    )

    # ============================================================
    # Round 1: 词源（应提取 pinned memory：词源定义）
    # ============================================================
    n1 = await cm.chat(
        user_message="御宅族这个词最早是怎么来的？它的词源和原始含义是什么？",
        parent_ids={n0},
        auto_plan=True,
    )

    # ============================================================
    # Round 2: 经典作品（应提取 pinned memory：里程碑作品列表）
    # ============================================================
    n2 = await cm.chat(
        user_message="1980年代到1990年代，有哪些里程碑式的动画作品推动了御宅族文化的发展？",
        parent_ids={n1},
        auto_plan=True,
    )

    # ============================================================
    # Round 3: 回到词源，测试 pinned memory 注入
    # ============================================================
    n3 = await cm.chat(
        user_message="基于你之前提到的词源，为什么'御宅'这个词从敬语变成了亚文化标签？",
        parent_ids={n2},
        auto_plan=True,
    )

    # ============================================================
    # Round 4: 互联网时代（继续拉长链，为压缩创造条件）
    # ============================================================
    n4 = await cm.chat(
        user_message="进入2000年后，互联网和轻小说如何改变了御宅族文化的生态？",
        parent_ids={n3},
        auto_plan=True,
    )

    # ============================================================
    # Round 5: 综合判断（此时链已足够长，应触发压缩）
    # ============================================================
    n5 = await cm.chat(
        user_message="综合以上所有讨论，你认为御宅族文化未来十年最可能向哪个方向发展？请结合你提取的关键记忆回答。",
        parent_ids={n4},
        auto_plan=True,
    )

    # ============================================================
    # 最终状态
    # ============================================================
    print_final_state(cm, latest_node_id=n5)


if __name__ == "__main__":
    asyncio.run(main())
