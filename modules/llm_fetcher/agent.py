import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from .llm_fetcher import LLMFetcher
from .llm_context import LLMContext, LLMContextHandler, LLMContextPair
from .tool import Tool, ToolRegistry

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]


class Agent:
    def __init__(
        self,
        llm_handler: LLMFetcher,
        system_prompt: str,
        tools: Optional[List[Any]] = None,
    ):
        self.llm_handler: LLMFetcher = llm_handler
        self._base_system_prompt: str = system_prompt
        self.memory_list: List[str] = []
        self.llm_context_hanlder = LLMContextHandler(llm_handler=self.llm_handler)
        self.tool_registry = ToolRegistry()

        # 注册内嵌工具（round_end 等），供 LLM 控制轮次生命周期
        self._register_builtin_tools()

        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

    def _register_builtin_tools(self) -> None:
        """注册 Agent 内嵌的元工具，用于控制对话轮次的生命周期。"""

        async def _round_end(**kwargs: Any) -> str:
            """结束当前 round_call。"""
            return "Round ended."

        self.tool_registry.register(
            Tool(
                name="round_end",
                description=(
                    "结束当前轮次。当你认为已经完成了本轮所有必要的思考、"
                    "工具调用和论点记录后，调用此工具来明确结束本轮对话。"
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=_round_end,
            )
        )

    @property
    def system_prompt(self) -> str:
        """Dynamic system prompt enriched with tool descriptions."""
        prompt = self._base_system_prompt
        # 当已通过 API 的 tools 参数注册原生工具时，不再在 prompt 中注入文本工具说明，
        # 避免 LLM 混淆：同时收到 "原生 function calling" 和 "文本 JSON" 两种指令。
        if not self.tool_registry.schemas:
            hint = self.tool_registry.get_prompt_hint()
            if hint:
                prompt = f"{prompt}\n{hint}"
        return prompt

    async def round_call(
        self,
        msg: str,
        stream: bool = False,
        verbose_info: bool = False,
        max_turns: int = 8
    ) -> str:
        """
        进行一整个轮次的 Agent 执行轮。

        核心特性：
        - 多轮工具调用循环：LLM 可在一次 round_call 内连续调用多个工具，
          每轮拿到工具结果后继续思考，直到决定结束。
        - 保留每轮 content：assistant 的 reasoning content 和 tool_calls
          都会保留在上下文中。
        - round_end：LLM 可通过调用 round_end() 主动结束本轮。

        执行流程：
        1. 构建完整 messages（system + 历史 + user msg）。
        2. 进入循环，最多 max_turns 轮：
           a. 调用 LLM（带 tools）。
           b. 若无 tool_calls → 直接结束。
           c. 若有 tool_calls → 执行工具，结果追加到 messages。
              - 若包含 round_end → 再调用一次 LLM（不带 tools）获取最终总结，结束。
           d. 继续下一轮。
        3. 将最终回复写入上下文。

        Args:
            msg: 本 agent 的本次输入。
            stream: 为 True 时，最终回复以逐字打印方式模拟流式输出到 stdout。
            verbose_info: 为 True 时，打印每轮调用、tool_calls、执行结果等调试信息。
            max_turns: 最大轮次。

        Returns:
            LLM 生成的完整回复文本。
        """
        # -------------------------------------------------
        # 1. 获取历史上下文，构建完整 messages
        # -------------------------------------------------
        prev_context = await self.llm_context_hanlder.get_now_context()

        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(prev_context)
        messages.append({"role": "user", "content": msg})

        turn_count = 0
        final_content = ""

        # -------------------------------------------------
        # 2. 多轮工具调用循环
        # -------------------------------------------------
        while turn_count < max_turns:
            turn_count += 1

            if verbose_info:
                print(f"\n[Agent] ====== 第 {turn_count} 轮调用 ======")

            response = await self.llm_handler.fetch(
                msg="",  # 空，因为所有消息已在 prev_messages 中
                system_prompt=None,  # system 已在 messages 中，避免重复
                prev_messages=messages,
                tools=self.tool_registry.schemas if self.tool_registry.schemas else None,
            )

            message = response.choices[0].message
            content = message.content or ""
            tool_calls = getattr(message, "tool_calls", None)

            if verbose_info:
                print(f"[Agent] content={content[:120]!r}...")
                print(f"[Agent] tool_calls={'有' if tool_calls else '无'}")

            # 若无 tool_calls，本轮自然结束
            if not tool_calls:
                final_content = content
                if verbose_info:
                    print("[Agent] 无 tool_calls，自然结束")
                break

            # -------------------------------------------------
            # 有 tool_calls：保留 assistant 消息（content + tool_calls）
            # -------------------------------------------------
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # 逐个执行工具
            has_round_end = False
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                # 结束本执行轮
                if tc.function.name == "round_end":
                    result = "Round ended."
                    has_round_end = True
                    if verbose_info:
                        print(f"[Agent] 内嵌工具 | round_end() -> {result}")
                else:
                # 或者执行其他工具
                    try:
                        result = await self.tool_registry.execute(tc.function.name, args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                    if verbose_info:
                        print(f"[Agent] 工具结果 | {tc.function.name} -> {str(result)[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

            # 若 LLM 主动调用 round_end，再调一次 LLM（不带 tools）获取最终总结
            if has_round_end:
                if verbose_info:
                    print("[Agent] round_end 触发，发起最终总结调用...")
                response_final = await self.llm_handler.fetch(
                    msg="",
                    system_prompt=None,
                    prev_messages=messages,
                    tools=None,  # 不提供 tools，强制让 LLM 输出最终回复
                )
                final_content = response_final.choices[0].message.content or ""
                break

        # -------------------------------------------------
        # 3. 将最终回复写入上下文
        # -------------------------------------------------
        await self.llm_context_hanlder.add_context(
            LLMContextPair(
                LLMContext(role="user", content=msg),
                LLMContext(role="assistant", content=final_content),
            )
        )

        # -------------------------------------------------
        # 4. stream 模式：逐字打印最终回复
        # -------------------------------------------------
        if stream and final_content:
            for char in final_content:
                print(char, end="", flush=True)
                await asyncio.sleep(0.0005)

        return final_content

    def extract_json_msg(self, msg: str) -> List[JSON]:
        """
        提取出一个 str 对象内所有的 JSON 对象。
        （保留作为 fallback，当原生 function calling 不可用时使用）

        Args:
            msg: 可能包含 JSON 片段的字符串。

        Returns:
            按出现顺序排列的所有顶层 JSON 对象（通常为 dict）列表。
        """
        results: List[JSON] = []
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(msg):
            # 跳过空白字符和非 JSON 起始字符
            while idx < len(msg) and msg[idx] not in "{[":
                idx += 1
            if idx >= len(msg):
                break
            try:
                obj, end = decoder.raw_decode(msg, idx)
                if isinstance(obj, dict):
                    results.append(obj)
                idx += end
            except (ValueError, json.JSONDecodeError):
                idx += 1
        return results

    async def tool_call(self, json_info: JSON) -> Optional[Dict[str, Any]]:
        """
        执行 JSON 格式的 tool call。
        保留作为 fallback，当 LLM 以文本 JSON 形式输出 tool call 时使用。
        期望格式: {"tool": "<name>", "arguments": {<key>: <value>, ...}}

        Args:
            json_info: 包含 tool 名称和参数的字典。

        Returns:
            {"tool": str, "result": Any} 或 None（解析失败时）。
        """
        if not isinstance(json_info, dict):
            return None

        tool_name = json_info.get("tool")
        if not tool_name or not isinstance(tool_name, str):
            return None

        arguments = json_info.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        try:
            result = await self.tool_registry.execute(tool_name, arguments)
        except Exception as exc:
            result = f"Error: {exc}"

        return {"tool": tool_name, "result": result}
