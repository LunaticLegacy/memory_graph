"""演示 RuntimeSlot —— 后台执行服务的用法。

场景：Agent 提交一个耗时 10s 的模拟任务，然后在后续轮次中轮询并收集结果。
"""

import asyncio

from modules import AgentSwarm, LLMFetcher, RuntimeSlotManager, Tool
from modules.llm_fetcher.swarm.runtime_slot_tools import create_runtime_slot_tools


async def demo_slot():
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key="sk-placeholder",
        model="deepseek-chat",
        timeout=60.0,
    )

    swarm = AgentSwarm(fetcher, name="slot_demo")

    # 定义一个耗时的模拟工具
    async def slow_crawl(**kwargs):
        url = kwargs.get("url", "")
        await asyncio.sleep(5)  # 模拟 5s 爬取
        return {"url": url, "title": "Example Page", "content": "..." * 100}

    crawl_tool = Tool(
        name="slow_crawl",
        description="模拟耗时网页爬取",
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
        handler=slow_crawl,
    )

    swarm.add_tool(crawl_tool)

    # 创建 slot manager 并注册其工具到 agent
    slot_mgr = RuntimeSlotManager(
        thinking_graph=swarm.thinking_graph,
        default_timeout=10.0,
        max_concurrent=2,
    )
    slot_tools = create_runtime_slot_tools(slot_mgr)
    for t in slot_tools:
        swarm.add_tool(t)

    # 提交后台任务（直接调用 manager，不经过 agent round）
    slot_id = await slot_mgr.submit(
        crawl_tool,
        {"url": "https://example.com"},
        name="crawl_example",
        timeout=8.0,
    )
    print(f"Submitted slot: {slot_id}")

    # 模拟轮询
    for i in range(10):
        slot = await slot_mgr.poll(slot_id)
        print(f"  Poll {i+1}: {slot.status.value} (poll_count={slot.poll_count})")
        if slot.status.value in ("completed", "failed", "timeout", "cancelled"):
            break
        await asyncio.sleep(1)

    # 收集结果
    if slot.status.value == "completed":
        result = await slot_mgr.collect(slot_id)
        print(f"Collected: {result}")
    else:
        print(f"Slot ended with status={slot.status.value}, error={slot.error}")

    # 查看 ThinkingGraph 中的记录
    tg = await swarm.thinking_graph.get_full_graph()
    print(f"\nThinkingGraph: {tg['node_count']} nodes, {tg['edge_count']} edges")
    for nid, node in tg["nodes"].items():
        print(f"  [{node['node_type']}] {node['info'][:60]}...")

    print(f"\nSlotManager snapshot: {slot_mgr.to_dict()}")


if __name__ == "__main__":
    asyncio.run(demo_slot())
