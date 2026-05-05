"""演示 AgentSwarm 的声明式构建与运行。

本文件展示 memory_graph 的 swarm 层级用法：
- 用 AgentSwarm 作为顶层容器
- 链式 API 声明式构建拓扑
- ThinkingGraph 作为 swarm 共享认知层
- 运行时动态修改（通过 ExecutionGraph tools）
"""

import asyncio
import os

from modules import (
    AgentSwarm,
    LLMFetcher,
    Tool,
)


async def get_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return api_key


async def demo_research_swarm():
    """示例：研究小组 Swarm —— 规划员 + 研究员 + 写作员。"""
    api_key = await get_api_key()
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    # 创建 swarm
    swarm = AgentSwarm(fetcher, name="research_team")

    # 添加全局工具（示例：一个简单的本地工具）
    async def word_count_tool(**kwargs):
        text = kwargs.get("text", "")
        return {"word_count": len(text.split())}

    swarm.add_tool(Tool(
        name="word_count",
        description="统计文本字数",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=word_count_tool,
    ))

    # 添加 agent 节点
    # share_thinking_tools=True 表示每个 agent 都能操作共享的 ThinkingGraph
    swarm.add_agent(
        "planner",
        "你是研究规划专家。请根据用户请求制定研究计划，用 thinking_graph_add_node 记录你的 plan 和 step。",
        share_thinking_tools=True,
    )
    swarm.add_agent(
        "researcher",
        "你是研究员。请执行研究计划，用 thinking_graph_add_node 记录你的 evidence 和 observation。",
        share_thinking_tools=True,
    )
    swarm.add_agent(
        "writer",
        "你是写作专家。请基于研究证据撰写总结报告，用 thinking_graph_add_node 记录你的 claim 和 summary。",
        share_thinking_tools=True,
    )

    # 添加 router：判断是否需要深入研究
    swarm.add_router(
        "router",
        routes={
            "deep": "需要深入研究",
            "shallow": "简单回答即可",
        },
        agent=None,  # 使用默认路由（这里简化演示）
        default_route="deep",
    )

    # 构建拓扑
    swarm.add_input("input")
    swarm.add_output("output")
    swarm.connect("input", "planner")
    swarm.connect("planner", "router")
    swarm.connect("router", "researcher", label="deep")
    swarm.connect("router", "writer", label="shallow")
    swarm.connect("researcher", "writer")
    swarm.connect("writer", "output")

    print("=" * 60)
    print("【Swarm 初始状态】")
    print("=" * 60)
    print(swarm.to_dict())

    # 运行
    topic = "人工智能对就业市场的影响"
    ctx = await swarm.run(initial_input=topic, entry_node_id="input")

    print("\n" + "=" * 60)
    print("【运行结果】")
    print("=" * 60)
    print(f"输出节点结果: {ctx.get_output('output')}")

    print("\n" + "=" * 60)
    print("【ThinkingGraph 共享认知状态】")
    print("=" * 60)
    tg = await swarm.thinking_graph.get_full_graph()
    print(f"节点数: {tg['node_count']}, 边数: {tg['edge_count']}")
    for nid, node in tg["nodes"].items():
        print(f"  [{node['node_type']}] {node['info'][:60]}...")

    print("\n" + "=" * 60)
    print("【Swarm 最终状态】")
    print("=" * 60)
    print(repr(swarm))


async def demo_dynamic_swarm():
    """示例：运行时动态增删 Agent 的 Swarm。"""
    api_key = await get_api_key()
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    swarm = AgentSwarm(fetcher, name="dynamic_team")

    # admin agent 拥有修改执行图的权限
    swarm.add_agent(
        "admin",
        "你是 swarm 管理员。你可以通过 tool_call 动态修改执行图结构：\n"
        "- execution_graph_add_agent: 添加新 Agent\n"
        "- execution_graph_connect: 连接节点\n"
        "- execution_graph_update_agent_prompt: 修改 Agent 提示词\n"
        "- execution_graph_get_info: 查看图结构",
        share_graph_tools=True,   # 允许修改 ExecutionGraph
        share_thinking_tools=True,
    )

    swarm.add_agent(
        "worker",
        "你是一位普通助手。请回答用户问题。",
        share_thinking_tools=True,
    )

    swarm.add_input("input")
    swarm.add_output("output")
    swarm.connect("input", "admin")
    swarm.connect("admin", "worker")
    swarm.connect("worker", "output")

    print("\n" + "=" * 60)
    print("【动态 Swarm — 初始状态】")
    print("=" * 60)
    print(swarm.to_dict())

    task = (
        "当前任务是：'请用诗歌形式回答'。\n"
        "请你先调用 execution_graph_update_agent_prompt，"
        "把 worker 节点的提示词修改为'你是一位诗人，用优美的诗歌回答所有问题'。\n"
        "然后再调用 execution_graph_get_info 查看当前图结构。"
    )
    ctx = await swarm.run(initial_input=task, entry_node_id="input")

    print("\n--- 运行后各节点输出 ---")
    for nid, out in ctx.node_outputs.items():
        print(f"[{nid}]: {str(out)[:300]}...")

    print("\n--- 运行后图结构 ---")
    print(swarm.execution_graph.to_dict())

    print("\n--- 验证动态修改效果 ---")
    print(f"worker system_prompt: {swarm.execution_graph.get_node('worker').agent.system_prompt[:80]}...")


async def main():
    await demo_research_swarm()
    # await demo_dynamic_swarm()


if __name__ == "__main__":
    asyncio.run(main())
