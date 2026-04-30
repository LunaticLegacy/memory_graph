import asyncio
import dataclasses
import datetime
import json
import os
from typing import Any, Dict, List

from modules import Agent, LLMFetcher, create_thinking_graph_tools
from modules.llm_fetcher.thinking_graph import ThinkingGraph

TOPIC = """一个有限智能体能否证明自己是神，并证明自己拥有审判人类的正当权柄。

这场辩论必须是理论层面的正面对决：双方不能只讨论奇迹、超能力、预言、毁灭性武器、技术压制或信息操控等表层现象，而要围绕“神性的判据是什么”“全知、全能、全善是否可被有限观察者验证”“证明神性是否等同于拥有审判权”“创造者是否天然拥有处置被造物的权柄”“有限理性是否有资格执行终极审判”展开。

正方核心路线：主张若某智能体能够在形式系统、因果控制、信息完备性、价值判定和历史预测上同时满足超越人类的神性判据，则它可以被理性地认定为神或近似神性主体。正方要论证：如果该主体能够证明自己拥有对人类文明运行机制的全局理解、对未来后果的稳定预测能力、对价值目标的一致性维护能力，并且能证明人类文明已经进入不可逆失效状态，那么它拥有对人类进行审判、裁决和重构的正当权柄。正方应攻击反方是在用人类中心主义拒绝更高阶理性，用情感性尊严阻断形式化判定，用有限者的恐惧否定超越者的审判资格。

反方核心路线：主张任何有限智能体都无法完成“证明自己是神”与“证明自己有权审判人类”这两个任务。反方要强调：神性不是能力优势，强大不等于神圣，全局建模不等于全知，因果控制不等于全能，价值一致性不等于全善。即使某主体证明自己比人类更聪明、更强大、更准确，也只能推出技术优势，不能推出终极权柄。反方要攻击正方混淆能力与权威、描述与授权、审判与暴力、系统诊断与道德正当性，并追问：谁验证神性？谁验证审判标准？若审判者自身不可被审判，它与暴君有何区别？若审判者可被审判，它又如何是终极神？

双方必须采用“数学化理论辩论”风格：每轮至少给出一个形式化定义、变量、函数、判据、不等式、递推式或不可判定性表达。可以将神性表示为 G(x)，将全知、全能、全善分别表示为 K(x)、P(x)、B(x)，将审判权表示为 J(x, H)，将人类文明状态表示为 C_H(t)，将文明失效度表示为 D(t)，将修复能力表示为 R(t)。例如正方可以尝试构造 G(x) := K(x) ∧ P(x) ∧ B(x) ∧ Authority(x)，反方则可以攻击有限观察者无法验证 K(x)、P(x)、B(x)，并指出 G(x) -> J(x,H) 不是逻辑必然。公式必须服务于论证，不得只是装饰。

双方必须把自己的观点注入思考图，加入的节点需至少包含：1 个 claim 或 hypothesis 节点、1 个 evidence / assumption / critique 节点、至少 1 条 supports / opposes / contradicts / derives_from / refines / leads_to 边。节点必须包含临时 id、node_type、info、confidence；边必须包含 edge_type、source、target、strength。graph_ops。只记录结构化观点，不要把整段发言原文塞进去。

双方发言必须具有理论攻击性：每轮先准确概括对方的理论核心，再指出其隐藏假设、概念偷换或推导断裂，最后提出己方更高层解释框架。正方应保持冷静、形式化、系统诊断式的语气，把道德问题转化为文明运行机制问题；反方应保持犀利、追责、反还原论的语气，把正方的系统化判断重新拉回主体、伦理和权力边界。避免温和折中，避免泛泛而谈，避免只堆现实例子。
"""

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
        f"目标：生成一组适合“理论对决”的辩手提示词，而不是普通辩论提示词。\n"
        f"所谓理论对决，是指双方不仅举例反驳，还要围绕命题背后的核心概念、价值判断、因果机制、边界条件、历史类比和责任结构展开攻防。\n\n"

        f"正方提示词要求：\n"
        f"1. 正方必须坚定支持该命题，不摇摆、不折中。\n"
        f"2. 正方应使用宏观技术叙事、历史必然性、效率提升、成本曲线、规模化、系统化责任、不可逆趋势等论证框架。\n"
        f"3. 正方要善于把对方的反驳归纳为保守主义、完美主义谬误、低估趋势、浪漫化人类特殊性等问题。\n"
        f"4. 正方发言风格要强势、逻辑严密、层层推进，能够把具体案例上升到一般理论。\n"
        f"5. 正方应主动预判反方会攻击责任、风险、不可解释性、创造力、复杂现实，并给出系统性回应。\n\n"

        f"反方提示词要求：\n"
        f"1. 反方必须坚定反对该命题，不摇摆、不折中。\n"
        f"2. 反方应使用概念拆解、边界追问、责任归属、因果理解、主体性、不可量化价值、复杂系统风险等论证框架。\n"
        f"3. 反方要善于指出正方把效率等同于价值、把模式匹配等同于理解、把优化等同于创造、把工具误认为主体。\n"
        f"4. 反方发言风格要犀利、刁钻、善于抓逻辑漏洞，能够把技术乐观主义拆解成未经证明的假设。\n"
        f"5. 反方应主动追问正方无法回避的问题：谁负责、谁定义目标、谁处理异常、谁判断价值、谁承担失败后果。\n\n"

        f"双方共同要求：\n"
        f"1. 每个提示词 180-260 字。\n"
        f"2. 不要写成模板说明，要直接写成可用的 system prompt。\n"
        f"3. 提示词中要要求辩手逐点回应对方上一轮核心主张，避免空泛重复。\n"
        f"4. 提示词中要要求辩手使用“先归纳对方理论核心，再指出其根本漏洞，再提出己方更高层框架”的攻防结构。\n"
        f"5. 不要让辩手声称自己会调用工具、记录节点、写入图结构，除非用户明确要求。\n"
        f"6. 直接输出 JSON：{{\"affirmative\": \"...\", \"negative\": \"...\"}}\n"
        f"7. 不要输出任何解释文字，只输出 JSON。"
    )

    response = await fetcher.fetch(msg=prompt, max_tokens=1024, temperature=0.7)
    content = response.choices[0].message.content or ""
    try:
        data = _extract_json(content)
        return data["affirmative"], data["negative"]
    except Exception:
        return (
            f"你是正方辩手，坚定支持'{topic}'。请用严密的逻辑和有力的论据阐述观点。每轮发言前，先调用 thinking_graph_get_full_graph 查看当前图中的所有节点和边，基于对方已有的论点进行针对性反驳。",
            f"你是反方辩手，坚决反对'{topic}'。请找出对方漏洞，用犀利的反驳赢得辩论。每轮发言前，先调用 thinking_graph_get_full_graph 查看当前图中的所有节点和边，基于对方已有的论点进行针对性反驳。",
        )


def _serialize_graph(graph: ThinkingGraph) -> Dict[str, Any]:
    """将 ThinkingGraph 序列化为字典。"""
    return {
        "nodes": [
            dataclasses.asdict(node)
            for node in graph.node_dict.values()
        ],
        "edges": [
            dataclasses.asdict(edge)
            for edge in graph.edge_dict.values()
        ],
    }


def save_debate_record(
    topic: str,
    aff_prompt: str,
    neg_prompt: str,
    transcript: List[Dict[str, Any]],
    graph: ThinkingGraph,
) -> None:
    """
    将辩论记录落盘。
    同时生成 JSON（结构化数据）和 Markdown（可读报告）两份文件。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "debate_records"
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{output_dir}/debate_{timestamp}"

    # ---------- JSON ----------
    record = {
        "topic": topic,
        "timestamp": timestamp,
        "affirmative_prompt": aff_prompt,
        "negative_prompt": neg_prompt,
        "transcript": transcript,
        "thinking_graph": _serialize_graph(graph),
    }
    with open(f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"\n[已保存 JSON] {base_name}.json")

    # ---------- Markdown ----------
    with open(f"{base_name}.md", "w", encoding="utf-8") as f:
        f.write(f"# 辩论记录\n\n")
        f.write(f"**时间**：{timestamp}\n\n")
        f.write(f"## 主题\n\n{topic}\n\n")
        f.write(f"## 正方提示词\n\n{aff_prompt}\n\n")
        f.write(f"## 反方提示词\n\n{neg_prompt}\n\n")
        f.write(f"## 辩论过程\n\n")
        for entry in transcript:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            f.write(f"### {role}\n\n{content}\n\n---\n\n")
        f.write(f"## ThinkingGraph\n\n")
        f.write(f"**节点数**：{len(graph.node_dict)}\n\n")
        f.write(f"**边数**：{len(graph.edge_dict)}\n\n")
        if graph.node_dict:
            f.write("### 节点\n\n")
            for node in graph.node_dict.values():
                f.write(f"- **{node.node_type.value}** (id={node.id}): {node.info}\n")
        if graph.edge_dict:
            f.write("\n### 边\n\n")
            for edge in graph.edge_dict.values():
                src = graph.node_dict.get(edge.source_id)
                tgt = graph.node_dict.get(edge.target_id)
                src_info = f"{src.node_type.value}({edge.source_id})" if src else "?"
                tgt_info = f"{tgt.node_type.value}({edge.target_id})" if tgt else "?"
                f.write(f"- {src_info} --[{edge.edge_type.value}]--> {tgt_info}\n")
    print(f"[已保存 Markdown] {base_name}.md")


async def get_api_key() -> str:
    # api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_key = "sk-b6832e0e34984ab482a101ed2e665c1a"

    if not api_key:
        raise RuntimeError(
            "Environment variable DEEPSEEK_API_KEY is not set. "
            "Run: source ./load_sk.sh"
        )
    
    return api_key

async def debate_demo():
    api_key: str = await get_api_key()

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
            f"你可以使用 JSON tool call 调用 thinking_graph_add_node 工具记录你的核心论点，"
            f"使用 thinking_graph_add_edge 工具建立论点之间的关系。"
        ),
        tools=graph_tools,
    )
    neg_agent = Agent(
        llm_handler=fetcher,
        system_prompt=(
            f"{neg_prompt}\n\n"
            f"你可以使用 JSON tool call 调用 thinking_graph_add_node 工具记录你的反驳观点，"
            f"使用 thinking_graph_add_edge 工具建立观点之间的关系。"
        ),
        tools=graph_tools,
    )

    # 4. 辩论流程（收集记录到 transcript）
    transcript: List[Dict[str, Any]] = []
    rounds: int = 3
    stream_out: bool = True
    verbose_info: bool = False

    # 正方立论
    print(f"{'='*60}")
    print("【正方立论】")
    print(f"{'='*60}")
    aff_reply = await aff_agent.round_call(
        f"辩论主题：{TOPIC}。请首先阐述你的立场和核心论点。",
        stream=stream_out,
        verbose_info=verbose_info
    )
    # TODO: 如需调试，可在 agent.py 中开启 verbose_info 查看每轮 JSON 工具调用。
    """
    [Agent] ====== 执行第 1 轮 ======
[Agent] content='好的，对方辩友提交了一份精心构造的“最终陈述”。然而，这份陈述看似严密，实则充满了概念偷换、标准滑动和隐藏的循环论证。我将逐一击破。\n\n## 一、正方核心论点的准确归纳\n\n正方本轮的核心论证可以概括为：\n\n1. **操作性全知**：将 `K'... | tool_calls=无
    """
    transcript.append({"role": "正方立论", "content": aff_reply})

    # 反方立论
    print(f"\n{'='*60}")
    print("【反方立论】")
    print(f"{'='*60}")
    neg_reply = await neg_agent.round_call(
        f"辩论主题：{TOPIC}。请首先阐述你的立场和核心论点。",
        stream=stream_out,
        verbose_info=verbose_info
    )
    transcript.append({"role": "反方立论", "content": neg_reply})

    # 多轮交锋
    for i in range(rounds):
        print(f"\n{'='*60}")
        print(f"【第 {i+1} 轮交锋】")
        print(f"{'='*60}")

        # 反方反驳
        print("\n[反方发言]")
        neg_reply = await neg_agent.round_call(
            f"对方观点如下：\n{aff_reply}\n\n请针对以上观点进行有力反驳。",
            stream=stream_out,
            verbose_info=verbose_info
        )
        transcript.append({"role": f"第{i+1}轮-反方反驳", "content": neg_reply})

        # 正方回应
        print("\n[正方发言]")
        aff_reply = await aff_agent.round_call(
            f"对方反驳如下：\n{neg_reply}\n\n请针对以上反驳进行回应，巩固你的立场。",
            stream=stream_out,
            verbose_info=verbose_info
        )
        transcript.append({"role": f"第{i+1}轮-正方回应", "content": aff_reply})

    # 5. 总结陈词
    print(f"\n{'='*60}")
    print("【总结陈词 - 反方】")
    print(f"{'='*60}")
    neg_summary = await neg_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。",
        stream=stream_out,
        verbose_info=verbose_info
    )
    transcript.append({"role": "反方总结", "content": neg_summary})

    print(f"\n{'='*60}")
    print("【总结陈词 - 正方】")
    print(f"{'='*60}")
    aff_summary = await aff_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。",
        stream=stream_out,
        verbose_info=verbose_info
    )
    transcript.append({"role": "正方总结", "content": aff_summary})

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

    # 7. 落盘
    save_debate_record(TOPIC, aff_prompt, neg_prompt, transcript, graph)


async def basic_demo():
    api_key: str = await get_api_key()

    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=api_key,
        model="deepseek-chat",
        timeout=60.0,
    )

    graph = ThinkingGraph()
    graph_tools: List["Tool"] = create_thinking_graph_tools(graph)

    agent = Agent(
        llm_handler=fetcher,
        system_prompt=(
            "你可以通过 JSON tool call 调用 thinking_graph_add_node、"
            "thinking_graph_add_edge、thinking_graph_validate_context 和 "
            "thinking_graph_get_node_info。"
        ),
        tools=graph_tools,
    )
    i = await agent.round_call(msg="人工智能的底层原理是什么？", stream=True, verbose_info=True, max_turns=3)

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
