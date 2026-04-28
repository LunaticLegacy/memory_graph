import asyncio
import os
from modules import LLMFetcher, LLMContextGraph, ContextManager


def print_graph_state(cm: ContextManager, highlight: int | None = None) -> None:
    print(f"\n{'-'*50}")
    print("上下文图快照")
    graph = cm.graph
    for nid, node in sorted(graph.nodes.items()):
        marker = ""
        if node.summarized_ids:
            marker += " [摘要节点]"
        if highlight is not None and nid == highlight:
            marker += " <-- 目标"
        print(f"  Node {nid}: parents={node.parent_ids}, children={node.child_ids}, summarized={node.summarized_ids}{marker}")

    print("\n关键记忆快照")
    memories = cm.memory_store.get_all_memories()
    if not memories:
        print("  （暂无）")
    else:
        for mem in memories:
            print(f"  Memory {mem.id} [tags={mem.tags}]: {mem.content[:80]}...")


async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable DEEPSEEK_API_KEY is not set. "
            "Run: `source ./load_sk.sh`."
            "If you don't have such file, create your own and add your own DEEPSEEK_API_KEY."
        )

    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    # 初始化上下文管理器
    cm = ContextManager(
        fetcher=fetcher,
        max_context_nodes=6,
        pack_keep_recent=1,
    )

    # ============================================================
    # Round 0: 建立基础认知
    # ============================================================
    n0 = await cm.chat(
        user_message="请用一段话介绍量子计算的核心思想。",
        auto_plan=False,  # 第一轮，图中无节点，跳过决策
    )

    # ============================================================
    # Round 1: 深入技术细节（Agent 应提取关键公式/概念到记忆）
    # ============================================================
    n1 = await cm.chat(
        user_message="量子比特的数学表示是什么？请写出叠加态和纠缠态的关键公式。",
        parent_ids={n0},
        auto_plan=True,
    )

    # ============================================================
    # Round 2: 换一个话题方向（Agent 应选择性加载上下文）
    # ============================================================
    n2 = await cm.chat(
        user_message="目前有哪些主要公司在做量子计算机？它们的技术路线有什么不同？",
        parent_ids={n1},
        auto_plan=True,
    )

    # ============================================================
    # Round 3: 回到核心概念，要求引用之前的公式（测试记忆提取）
    # ============================================================
    n3 = await cm.chat(
        user_message="基于你之前提到的量子比特叠加态公式，如果测量一个处于叠加态的量子比特，会发生什么？",
        parent_ids={n2},
        auto_plan=True,
    )

    # ============================================================
    # Round 4: 长对话后的压缩测试（Agent 应决定压缩旧节点）
    # ============================================================
    n4 = await cm.chat(
        user_message="乐观估计，还要多久才能实现大规模容错量子计算？",
        parent_ids={n3},
        auto_plan=True,
    )

    # ============================================================
    # Round 5: 综合判断（测试记忆 + 压缩后的上下文）
    # ============================================================
    n5 = await cm.chat(
        user_message="综合以上所有讨论，你认为量子计算未来十年最可能率先颠覆哪个行业？请结合你提取的关键记忆回答。",
        parent_ids={n4},
        auto_plan=True,
    )

    # ============================================================
    # 最终状态打印
    # ============================================================
    print("\n" + "="*50)
    print("最终状态总结")
    print("="*50)
    print_graph_state(cm)
    print(f"\nn5 的最长祖先链: {cm.graph.get_ancestor_chain(n5, max_nodes=8, strategy='longest')}")
    print(f"n5 的所有祖先: {cm.graph.get_all_ancestors(n5)}")


if __name__ == "__main__":
    asyncio.run(main())
