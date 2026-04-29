import asyncio
import os
from modules import (LLMFetcher, Agent)

async def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable DEEPSEEK_API_KEY is not set. "
            "Run: source ./load_sk.sh"
        )
    
    # 创建一个 LLMFetcher，并以此为基础创建一个 Agent
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    agent: Agent = Agent(
        llm_handler=fetcher,
        system_prompt="""
You are the coder major in Python.
        """
    )

    # 执行一整个大轮次的执行轮，并且，需要告知本 agent 的工作空间地址。
    await agent.round_call("Please run at:")


if __name__ == "__main__":
    asyncio.run(main())
