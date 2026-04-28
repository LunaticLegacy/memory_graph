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
    # Round 0
    # ============================================================
    n0 = await cm.chat(
        user_message="请用一段话介绍量子计算的核心思想。",
        auto_plan=False,
    )

    # ============================================================
    # Round 1: 公式（应提取 pinned memory）
    # ============================================================
    n1 = await cm.chat(
        user_message="量子比特的数学表示是什么？请写出叠加态和纠缠态的关键公式。",
        parent_ids={n0},
        auto_plan=True,
    )

    # ============================================================
    # Round 2: 产业现状
    # ============================================================
    n2 = await cm.chat(
        user_message="目前有哪些主要公司在做量子计算机？它们的技术路线有什么不同？",
        parent_ids={n1},
        auto_plan=True,
    )

    # ============================================================
    # Round 3: 回到公式，测试 pinned memory 注入
    # ============================================================
    n3 = await cm.chat(
        user_message="基于你之前提到的叠加态公式，如果测量一个处于叠加态的量子比特，会发生什么？",
        parent_ids={n2},
        auto_plan=True,
    )

    # ============================================================
    # Round 4: 时间线（继续拉长链，为压缩创造条件）
    # ============================================================
    n4 = await cm.chat(
        user_message="乐观估计，还要多久才能实现大规模容错量子计算？",
        parent_ids={n3},
        auto_plan=True,
    )

    # ============================================================
    # Round 5: 综合判断（此时链已足够长，应触发压缩）
    # ============================================================
    n5 = await cm.chat(
        user_message="综合以上所有讨论，你认为量子计算未来十年最可能率先颠覆哪个行业？请结合你提取的关键记忆回答。",
        parent_ids={n4},
        auto_plan=True,
    )

    # ============================================================
    # 最终状态
    # ============================================================
    print_final_state(cm, latest_node_id=n5)


if __name__ == "__main__":
    asyncio.run(main())
