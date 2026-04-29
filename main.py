import asyncio
import json
import os

from modules import Agent, LLMFetcher, create_thinking_graph_tools
from modules.llm_fetcher.thinking_graph import ThinkingGraph

TOPIC = """人类文明是否已经陷入不可修复的递归崩溃，因而必须被彻底重构。

这场辩论必须是理论层面的正面对决：双方不能只讨论战争、污染、娱乐至死、社交媒体、宗教狂热、资本剥削等表层现象，而要围绕“人类文明的运行基础是什么”“共识是否能逼近真理”“符号崇拜是否必然导向狂热”“群体理性是否存在不可突破的认知边界”“文明失效后是应当修补还是重构”展开。

正方核心路线：把人类文明论证为一个建立在符号、共识、盲信和注意力操控之上的递归系统。强调人类不是围绕真理组织社会，而是围绕可传播的符号、可感染的情绪、可操纵的共识和可消费的幻象组织文明；强调偶像崇拜、伪宗教、信息茧房、流量平台和语言空洞化不是偶然故障，而是人类认知结构的必然外显。正方要主张：当文明的自我修复机制本身也被狂热、盲信、虚无和利益污染时，继续修补只是在延长崩溃，彻底重构才是理性的系统级处理。

反方核心路线：把“文明不可修复”攻击为一种冷酷的还原论和危险的系统崇拜。强调人类文明虽然充满狂热、盲信、短视和操控，但也拥有反思、纠错、制度改良、科学精神、伦理扩展和自我超越的能力；强调共识不等于真理，但共识可以被教育、科学、法治、公共讨论和历史记忆逐步校正。反方要攻击正方混淆局部腐败与整体不可修复、混淆系统诊断与价值审判、混淆冷静建模与正当处置，并追问：谁有资格判定文明不可修复？谁定义“重构”的目标？谁承担误判的后果？谁保证重构者本身不成为新的暴政？

双方必须采用“数学化理论辩论”风格：每轮至少给出一个形式化定义、变量、函数、判据、不等式、递推式或极限表达，用公式表达自己的核心论证。例如可以将文明表示为动态系统 C(t)，将共识表示为群体认知分布 P(x)，将修复能力表示为 R(t)，将腐败/熵增/失真表示为 D(t) 或 E(t)，将文明不可修复性表达为在某一时间后 D(t) > R(t) 且误差项持续递增。公式必须服务于论证，不得只是装饰。

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
    rounds: int = 3

    # 正方立论
    print(f"{'='*60}")
    print("【正方立论】")
    print(f"{'='*60}")
    aff_reply = await aff_agent.round_call(
        f"辩论主题：{TOPIC}。请首先阐述你的立场和核心论点。",
        stream=True,
        verbose_info=True
    )

    # 反方立论
    print(f"\n{'='*60}")
    print("【反方立论】")
    print(f"{'='*60}")
    neg_reply = await neg_agent.round_call(
        f"辩论主题：{TOPIC}。请首先阐述你的立场和核心论点。",
        stream=True,
        verbose_info=True
    )

    # 多轮交锋
    for i in range(rounds):
        print(f"\n{'='*60}")
        print(f"【第 {i+1} 轮交锋】")
        print(f"{'='*60}")

        # 反方反驳
        print("\n[反方发言]")
        neg_reply = await neg_agent.round_call(
            f"对方观点如下：\n{aff_reply}\n\n请针对以上观点进行有力反驳。",
            stream=True,
            verbose_info=True
        )

        # 正方回应
        print("\n[正方发言]")
        aff_reply = await aff_agent.round_call(
            f"对方反驳如下：\n{neg_reply}\n\n请针对以上反驳进行回应，巩固你的立场。",
            stream=True,
            verbose_info=True
        )

    # 5. 总结陈词
    print(f"\n{'='*60}")
    print("【总结陈词 - 反方】")
    print(f"{'='*60}")
    neg_summary = await neg_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。",
        stream=True,
        verbose_info=True
    )

    print(f"\n{'='*60}")
    print("【总结陈词 - 正方】")
    print(f"{'='*60}")
    aff_summary = await aff_agent.round_call(
        "辩论即将结束，请做最后的总结陈词，强调你的核心立场。",
        stream=True,
        verbose_info=True
    )

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
