"""
ExecutionGraph 使用示例：演示运行时动态增删 Agent / 工具 / 修改提示词。

执行图结构示例：
  [input] --> [agent_A] --> [router] --[summarize]--> [agent_B]
                            --[expand]----> [agent_C] --> [output]

在运行过程中，agent_A 可以通过 tool_call 动态：
1. 添加新 agent 到图中
2. 给其他 agent 添加/删除工具
3. 修改其他 agent 的提示词
4. 连接/断开节点
"""

import asyncio
import os

from modules import (
    Agent,
    ExecutionGraph,
    LLMFetcher,
    Tool,
    create_execution_graph_tools,
    create_thinking_graph_tools,
)
from modules.llm_fetcher.thinking_graph import ThinkingGraph


async def get_api_key() -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return api_key


async def demo_static_graph():
    """示例 1：构建并运行一个静态执行图（无运行时修改）。"""
    api_key = await get_api_key()
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    graph = ExecutionGraph(llm_fetcher=fetcher)

    # 创建共享的 ThinkingGraph 工具
    thinking_graph = ThinkingGraph()
    tg_tools = create_thinking_graph_tools(thinking_graph)

    # 创建 agent_A：负责分析主题
    agent_a = Agent(
        llm_handler=fetcher,
        system_prompt="你是一位分析专家。请分析用户输入的主题，提炼核心观点。",
        tools=tg_tools,
    )
    node_a = graph.add_agent_node(agent_a, node_id="analyzer")

    # 创建 router：判断分析结果是偏"概括"还是"扩展"
    node_router = graph.add_router_node(
        routes={
            "summarize": "用户需要概括总结",
            "expand": "用户需要深入扩展",
        },
        agent=Agent(
            llm_handler=fetcher,
            system_prompt="你是一个路由决策器。请根据输入判断需要'summarize'还是'expand'，只输出标签。",
        ),
        node_id="router",
    )

    # 创建 agent_B：概括专家
    agent_b = Agent(
        llm_handler=fetcher,
        system_prompt="你是一位概括专家。请将输入内容精炼为 3 句话以内的总结。",
    )
    node_b = graph.add_agent_node(agent_b, node_id="summarizer")

    # 创建 agent_C：扩展专家
    agent_c = Agent(
        llm_handler=fetcher,
        system_prompt="你是一位扩展专家。请对输入内容进行深入扩展，补充细节和例子。",
    )
    node_c = graph.add_agent_node(agent_c, node_id="expander")

    # 出口节点
    node_out = graph.add_output_node(
        collector=lambda inputs: "\n\n--- 最终输出 ---\n\n".join(str(i) for i in inputs),
        node_id="output",
    )

    # 入口节点
    node_in = graph.add_input_node(node_id="input")

    # 连接图
    graph.connect("input", "analyzer")
    graph.connect("analyzer", "router")
    graph.connect("router", "summarizer", label="summarize")
    graph.connect("router", "expander", label="expand")
    graph.connect("summarizer", "output")
    graph.connect("expander", "output")

    # 运行
    print("=" * 60)
    print("【静态执行图示例】")
    print("=" * 60)

    topic = "人工智能对就业市场的影响"
    ctx = await graph.run(initial_input=topic, entry_node_id="input")

    print("\n--- 各节点输出 ---")
    for nid, out in ctx.node_outputs.items():
        print(f"[{nid}]: {str(out)[:200]}...")

    final = ctx.get_output("output")
    print(f"\n--- 最终汇聚结果 ---\n{final}")


async def demo_dynamic_graph():
    """示例 2：运行时动态修改执行图。"""
    api_key = await get_api_key()
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    graph = ExecutionGraph(llm_fetcher=fetcher)

    # 先把执行图自身的操作工具注册到全局工具池
    eg_tools = create_execution_graph_tools(graph)
    for tool in eg_tools:
        graph.register_tool(tool)

    # 创建"管理员 Agent"，它拥有修改执行图的能力
    admin_agent = Agent(
        llm_handler=fetcher,
        system_prompt=(
            "你是执行图的管理员。你可以通过 tool_call 动态修改执行图结构：\n"
            "- execution_graph_add_agent: 添加新 Agent\n"
            "- execution_graph_remove_node: 删除节点\n"
            "- execution_graph_connect: 连接节点\n"
            "- execution_graph_update_agent_prompt: 修改 Agent 提示词\n"
            "- execution_graph_add_tool_to_agent: 给 Agent 添加工具\n"
            "- execution_graph_remove_tool_from_agent: 给 Agent 移除工具\n"
            "- execution_graph_get_info: 查看图结构\n"
        ),
        tools=eg_tools,
    )

    # 创建"普通 Agent"，初始没有任何特殊能力
    worker_agent = Agent(
        llm_handler=fetcher,
        system_prompt="你是一位普通助手。请回答用户问题。",
    )

    admin_node = graph.add_agent_node(admin_agent, node_id="admin")
    worker_node = graph.add_agent_node(worker_agent, node_id="worker")

    # 入口 -> admin -> worker -> 输出
    node_in = graph.add_input_node(node_id="input")
    node_out = graph.add_output_node(node_id="output")

    graph.connect("input", "admin")
    graph.connect("admin", "worker")
    graph.connect("worker", "output")

    print("\n" + "=" * 60)
    print("【动态执行图示例】")
    print("=" * 60)
    print("初始图结构：", graph.to_dict())

    # 第一轮：admin 收到任务，可能决定修改 worker 的提示词
    task = (
        "当前任务是：'请用诗歌形式回答'。\n"
        "请你先调用 execution_graph_update_agent_prompt，"
        "把 worker 节点的提示词修改为'你是一位诗人，用优美的诗歌回答所有问题'。\n"
        "然后再调用 execution_graph_get_info 查看当前图结构。"
    )
    ctx = await graph.run(initial_input=task, entry_node_id="input")

    print("\n--- 第一轮执行后各节点输出 ---")
    for nid, out in ctx.node_outputs.items():
        print(f"[{nid}]: {str(out)[:300]}...")

    print("\n--- 第一轮后图结构 ---")
    print(graph.to_dict())

    # 第二轮：验证 worker 的提示词是否已被修改
    print("\n--- 验证动态修改效果 ---")
    print(f"worker system_prompt: {graph.get_node('worker').agent.system_prompt[:80]}...")


async def demo_simple_pipeline():
    """示例 3：最简单的线性流水线（不需要 LLM 推理的纯工具演示）。"""
    from modules.llm_fetcher.tool import ToolRegistry

    # 定义一个纯本地工具
    async def uppercase_tool(**kwargs):
        text = kwargs.get("input", "")
        return text.upper()

    async def reverse_tool(**kwargs):
        text = kwargs.get("input", "")
        return text[::-1]

    tool_upper = Tool(
        name="uppercase",
        description="将文本转为大写",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        handler=uppercase_tool,
    )
    tool_reverse = Tool(
        name="reverse",
        description="将文本反转",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        handler=reverse_tool,
    )

    graph = ExecutionGraph()
    graph.register_tool(tool_upper)
    graph.register_tool(tool_reverse)

    node_in = graph.add_input_node(node_id="input")
    node_upper = graph.add_tool_node(tool_upper, node_id="upper")
    node_reverse = graph.add_tool_node(tool_reverse, node_id="reverse")
    node_out = graph.add_output_node(node_id="output")

    graph.connect("input", "upper")
    graph.connect("upper", "reverse")
    graph.connect("reverse", "output")

    print("\n" + "=" * 60)
    print("【纯工具流水线示例】")
    print("=" * 60)

    ctx = await graph.run(initial_input="Hello ExecutionGraph!", entry_node_id="input")
    print(f"最终输出: {ctx.get_output('output')}")

    # 运行时动态修改：断开 reverse，直接输出 upper 的结果
    print("\n--- 运行时断开 reverse 节点 ---")
    graph.disconnect("upper", "reverse")
    graph.disconnect("reverse", "output")
    graph.connect("upper", "output")

    ctx2 = await graph.run(initial_input="Dynamic Graph!", entry_node_id="input")
    print(f"修改后输出: {ctx2.get_output('output')}")


async def demo_parallel_pipeline():
    """示例 4：并行流水线 + 汇聚 + 并发控制 + 超时演示。"""
    from modules.llm_fetcher.tool import Tool

    async def slow_fetch(**kwargs):
        """模拟耗时 0.3s 的 IO 操作。"""
        await asyncio.sleep(0.3)
        return f"fetched: {kwargs.get('input', '')}"

    async def slow_compute(**kwargs):
        """模拟耗时 0.3s 的计算。"""
        await asyncio.sleep(0.3)
        return f"computed: {kwargs.get('input', '')}"

    def merge(**kwargs):
        inputs = kwargs.get("inputs", [])
        return {"merged": inputs}

    t_fetch = Tool(
        name="fetch", description="模拟数据获取",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        handler=slow_fetch,
    )
    t_compute = Tool(
        name="compute", description="模拟计算",
        parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
        handler=slow_compute,
    )
    t_merge = Tool(
        name="merge", description="合并结果",
        parameters={"type": "object", "properties": {"inputs": {"type": "array"}}, "required": ["inputs"]},
        handler=merge,
    )

    # 场景 1：无并发限制，fetch 和 compute 并行执行
    print("\n" + "=" * 60)
    print("【并行流水线 - 无并发限制】")
    print("=" * 60)

    graph = ExecutionGraph()
    graph.add_input_node("in")
    graph.add_tool_node(t_fetch, "fetcher")
    graph.add_tool_node(t_compute, "computer")
    graph.add_tool_node(t_merge, "merger")
    graph.add_output_node(node_id="out")

    graph.connect("in", "fetcher")
    graph.connect("in", "computer")
    graph.connect("fetcher", "merger")
    graph.connect("computer", "merger")
    graph.connect("merger", "out")

    start = asyncio.get_event_loop().time()
    ctx = await graph.run("data", "in")
    elapsed = asyncio.get_event_loop().time() - start

    print(f"总耗时: {elapsed:.2f}s (两个 0.3s 分支并行，理论最小 0.3s)")
    print(f"输出: {ctx.get_output('out')}")

    # 场景 2：并发限制为 1，fetch 和 compute 串行
    print("\n" + "=" * 60)
    print("【并行流水线 - max_concurrency=1】")
    print("=" * 60)

    graph2 = ExecutionGraph(max_concurrency=1)
    graph2.add_input_node("in")
    graph2.add_tool_node(t_fetch, "fetcher")
    graph2.add_tool_node(t_compute, "computer")
    graph2.add_tool_node(t_merge, "merger")
    graph2.add_output_node(node_id="out")

    graph2.connect("in", "fetcher")
    graph2.connect("in", "computer")
    graph2.connect("fetcher", "merger")
    graph2.connect("computer", "merger")
    graph2.connect("merger", "out")

    start = asyncio.get_event_loop().time()
    ctx2 = await graph2.run("data", "in")
    elapsed = asyncio.get_event_loop().time() - start

    print(f"总耗时: {elapsed:.2f}s (串行执行，理论 0.6s)")
    print(f"输出: {ctx2.get_output('out')}")

    # 场景 3：节点超时
    print("\n" + "=" * 60)
    print("【节点超时演示】")
    print("=" * 60)

    async def forever(**kwargs):
        await asyncio.sleep(100)
        return "never"

    t_slow = Tool(
        name="forever", description="永远跑不完",
        parameters={"type": "object", "properties": {}},
        handler=forever,
    )

    graph3 = ExecutionGraph()
    graph3.add_input_node("in")
    graph3.add_tool_node(t_slow, "slow")
    graph3.add_output_node(node_id="out")
    graph3.connect("in", "slow")
    graph3.connect("slow", "out")
    graph3.set_node_timeout("slow", 0.2)

    start = asyncio.get_event_loop().time()
    ctx3 = await graph3.run("data", "in")
    elapsed = asyncio.get_event_loop().time() - start

    print(f"总耗时: {elapsed:.2f}s (超时设为 0.2s)")
    print(f"slow 节点输出: {ctx3.get_output('slow')}")

    # 场景 4：使用 JoinNode 显式汇聚
    print("\n" + "=" * 60)
    print("【JoinNode 显式汇聚】")
    print("=" * 60)

    graph4 = ExecutionGraph()
    graph4.add_input_node("in")
    graph4.add_tool_node(t_fetch, "fetcher")
    graph4.add_tool_node(t_compute, "computer")
    graph4.add_join_node(strategy="all", node_id="join")
    graph4.add_output_node(node_id="out")

    graph4.connect("in", "fetcher")
    graph4.connect("in", "computer")
    graph4.connect("fetcher", "join")
    graph4.connect("computer", "join")
    graph4.connect("join", "out")

    ctx4 = await graph4.run("data", "in")
    print(f"JoinNode 输出: {ctx4.get_output('join')}")
    print(f"最终输出: {ctx4.get_output('out')}")


async def main():
    # 按需取消注释以运行不同示例
    await demo_simple_pipeline()
    await demo_static_graph()
    await demo_dynamic_graph()
    await demo_parallel_pipeline()


if __name__ == "__main__":
    asyncio.run(main())
