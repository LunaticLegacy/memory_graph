import asyncio
import json
import os

from modules import Agent, LLMFetcher, create_thinking_graph_tools
from modules.llm_fetcher.thinking_graph import ThinkingGraph


TOPIC = "人工智能是否会取代人类程序员"


def _extract_json(text: str) -> dict:
    """从可能包含 markdown 的文本中提取 JSON 对象。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


async def generate_debate_prompts(fetcher: LLMFetcher, topic: str) -> tuple[str, str]:
    """让 LLM 生成正反方的系统提示词。"""
    prompt = (
        f'请为辩论主题"{topic}"生成正反双方的系统提示词（system prompt）。\n\n'
        f"要求：\n"
        f"1. 正方提示词：设定为正方辩手，支持该命题，语气坚定、逻辑严密、善于举例\n"
        f"2. 反方提示词：设定为反方辩手，反对该命题，善于找漏洞、反驳有力、角度刁钻\n"
        f"3. 每个提示词 150-250 字\n"
        f'4. 直接输出 JSON：{{"affirmative": "...", "negative": "..."}}\n'
        f"5. 不要输出任何解释文字，只输出 JSON"
    )

    response = await fetcher.fetch(msg=prompt, max_tokens=1024, temperature=0.7)
    content = response.choices[0].message.content or ""
    try:
        data = _extract_json(content)
        return data["affirmative"], data["negative"]
    except Exception:
        return (
            f"你是正方辩手，坚定支持'{topic}'。请用严密的逻辑和有力的论据阐述观点。",
            f"你是反方辩手，坚决反对'{topic}'。请找出对方漏洞，用犀利的反驳赢得辩论。",
        )


async def debate_demo():
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

    # 1. 生成提示词
    print(f"{'='*60}")
    print(f"辩论主题：{TOPIC}")
    print(f"{'='*60}\n")

    print("[正在生成正反方角色提示词...]")
    aff_prompt, neg_prompt = await generate_debate_prompts(fetcher, TOPIC)

    print(f"\n--- 正方提示词 ---\n{aff_prompt}\n")
    print(f"--- 反方提示词 ---\n{neg_prompt}\n")

    # 2. 创建 ThinkingGraph（共享给双方）
    graph = ThinkingGraph()
    graph_tools = create_thinking_graph_tools(graph)

    # 3. 创建两个 Agent
    aff_agent = Agent(
        llm_handler=fetcher,
        system_prompt=(
            f"{aff_prompt}\n\n"
            f"你可以使用 graph_add_node 工具记录你的核心论点，"
            f"使用 graph_add_edge 工具建立论点之间的关系。"
        ),
        tools=graph_tools,
    )
    neg_agent = Agent(
        llm_handler=fetcher,
        system_prompt=(
            f"{neg_prompt}\n\n"
            f"你可以使用 graph_add_node 工具记录你的反驳观点，"
            f"使用 graph_add_edge 工具建立观点之间的关系。"
        ),
        tools=graph_tools,
    )

    # 4. 辩论流程
    rounds = 3

    # 正方立论
    print(f"{'='*60}")
    print("【正方立论】")
    print(f"{'='*60}")
    aff_reply = await aff_agent.round_call(
        f"辩论主题：{TOPIC}。请首先阐述你的立场和核心论点。"
    )
    print(aff_reply)

    # 多轮交锋
    for i in range(rounds):
        print(f"\n{'='*60}")
        print(f"【第 {i+1} 轮交锋】")
        print(f"{'='*60}")

        # 反方反驳
        print("\n[反方发言]")
        neg_reply = await neg_agent.round_call(
            f"对方观点如下：\n{aff_reply}\n\n请针对以上观点进行有力反驳。"
        )
        print(neg_reply)

        # 正方回应
        print("\n[正方发言]")
        aff_reply = await aff_agent.round_call(
            f"对方反驳如下：\n{neg_reply}\n\n请针对以上反驳进行回应，巩固你的立场。"
        )
        print(aff_reply)

    # 5. 总结陈词
    print(f"\n{'='*60}")
    print("【总结陈词 - 反方】")
    print(f"{'='*60}")
    neg_summary = await neg_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。"
    )
    print(neg_summary)

    print(f"\n{'='*60}")
    print("【总结陈词 - 正方】")
    print(f"{'='*60}")
    aff_summary = await aff_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。"
    )
    print(aff_summary)

    # 6. 展示 thinking_graph 状态
    print(f"\n{'='*60}")
    print(f"【ThinkingGraph 统计】")
    print(f"{'='*60}")
    print(f"节点数：{len(graph.node_dict)}")
    print(f"边数：{len(graph.edge_dict)}")

    if graph.node_dict:
        print("\n--- 节点列表 ---")
        for node in graph.node_dict.values():
            print(f"  [{node.node_type.value}] {node.info[:60]}...")

    if graph.edge_dict:
        print("\n--- 边列表 ---")
        for edge in graph.edge_dict.values():
            src = graph.node_dict.get(edge.source_id)
            tgt = graph.node_dict.get(edge.target_id)
            src_type = src.node_type.value if src else "?"
            tgt_type = tgt.node_type.value if tgt else "?"
            print(f"  {src_type} -[{edge.edge_type.value}]-> {tgt_type}")


if __name__ == "__main__":
    asyncio.run(debate_demo())
