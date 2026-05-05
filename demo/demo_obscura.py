"""演示 Obscura Tool 用法 —— 在 AgentSwarm 中调用 headless browser 获取网页内容。"""

import asyncio
import os

from modules import AgentSwarm, LLMFetcher, create_obscura_tools


async def get_api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "")


async def demo_web_fetch():
    """示例 1：直接用 Tool 调用 obscura fetch（不经过 LLM）。"""
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=await get_api_key(),
        model="deepseek-chat",
        timeout=60.0,
    )

    swarm = AgentSwarm(fetcher, name="obscura_demo")
    for t in create_obscura_tools():
        swarm.add_tool(t)

    # 直接调用 Tool（不走 Agent round_call）
    from modules.llm_fetcher.tools.obscura_tools import _obscura_fetch_cli

    result = await _obscura_fetch_cli(
        url="https://example.com",
        mode="text",
        wait=2,
    )
    print("=" * 60)
    print("【直接调用 web_fetch】")
    print("=" * 60)
    print(f"exit_code: {result['exit_code']}")
    print(f"stdout:\n{result['stdout'][:500]}...")


async def demo_agent_with_obscura():
    """示例 2：Agent 调用 web_fetch Tool 获取网页。"""
    api_key = await get_api_key()
    if not api_key:
        print("DEEPSEEK_API_KEY not set, skipping agent demo.")
        return

    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    swarm = AgentSwarm(fetcher, name="obscura_agent")
    for t in create_obscura_tools():
        swarm.add_tool(t)

    swarm.add_agent(
        "browser_agent",
        "你是一个网页分析助手。你可以使用 web_fetch 工具获取网页内容并进行分析。"
        "当用户要求查看网页时，调用 web_fetch 获取文本内容后给出摘要。",
        share_thinking_tools=True,
    )

    swarm.add_input("input")
    swarm.add_output("output")
    swarm.connect("input", "browser_agent")
    swarm.connect("browser_agent", "output")

    print("\n" + "=" * 60)
    print("【Agent 调用 obscura】")
    print("=" * 60)

    ctx = await swarm.run(
        initial_input="请获取 https://example.com 的内容并简要描述这个网站是做什么的。",
        entry_node_id="input",
    )
    print(f"\n输出: {ctx.get_output('output')}")


async def main():
    await demo_web_fetch()
    # await demo_agent_with_obscura()


if __name__ == "__main__":
    asyncio.run(main())
