import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from .llm_fetcher import LLMFetcher
from .llm_context import LLMContext, LLMContextHandler, LLMContextPair
from .tool import ToolRegistry

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

        if tools:
            for tool in tools:
                self.tool_registry.register(tool)

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
    ) -> str:
        """
        进行一整个轮次的 Agent 执行轮。
        支持原生 Function Calling：若 LLM 返回 tool_calls，则执行工具并将结果回传。

        执行流程：
        1. 获取当前上下文历史（不包含本次 msg）。
        2. 第一轮 LLM 调用（非流式，传入 tools schema），检测 tool_calls。
        3. 若 LLM 请求了 tool_calls：
           - 执行每个工具调用；
           - 构建包含 tool results 的第二轮消息；
           - 第二轮调用获取最终回复（stream=True 时流式输出到 stdout）。
        4. 若 LLM 未请求 tool_calls：
           - 直接使用第一轮返回的文本；
           - stream=True 时以逐字打印方式模拟流式效果。
        5. 将最终回复写入上下文。

        Args:
            msg: 本 agent 的本次输入。
            stream: 为 True 时，最终回复以流式方式输出到 stdout。
            verbose_info: 为 True 时，打印 tool_calls 检测、执行参数与结果等调试信息。

        Returns:
            LLM 生成的完整回复文本。
        """
        # -------------------------------------------------
        # 1. 获取当前上下文历史（不包含本次 msg）
        # -------------------------------------------------
        prev_context = await self.llm_context_hanlder.get_now_context()

        # -------------------------------------------------
        # 2. 第一轮 LLM 调用：非流式，传入 tools，检测 tool_calls
        # -------------------------------------------------
        tools_schema = self.tool_registry.schemas
        if verbose_info:
            print(f"\n[Agent] 第一轮调用 | tools={'启用' if tools_schema else '未启用'}")

        response = await self.llm_handler.fetch(
            msg=msg,
            system_prompt=self.system_prompt,
            prev_messages=prev_context,
            tools=tools_schema if tools_schema else None,
        )

        message = response.choices[0].message
        content = message.content or ""

        # -------------------------------------------------
        # 3. 若 LLM 请求了原生 tool_calls，执行工具并发起第二轮调用
        # -------------------------------------------------
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            if verbose_info:
                print(f"[Agent] 检测到 {len(tool_calls)} 个 tool_call(s)")
                for tc in tool_calls:
                    print(f"  -> {tc.function.name}({tc.function.arguments})")

            # 构建第二次调用的对话历史
            second_messages = prev_context.copy()
            second_messages.append({"role": "user", "content": msg})
            second_messages.append({
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

            # 逐个执行工具调用
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                try:
                    result = await self.tool_registry.execute(tc.function.name, args)
                except Exception as exc:
                    result = f"Error: {exc}"
                second_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
                if verbose_info:
                    print(f"[Agent] 工具结果 | {tc.function.name} -> {str(result)[:200]}")

            # 第二轮调用获取最终回复
            if stream:
                if verbose_info:
                    print("[Agent] 第二轮调用（流式输出）...\n")
                out_msg = ""
                async for chunk in self.llm_handler.fetch_stream(
                    msg="请基于工具返回的结果继续回答。",
                    system_prompt=self.system_prompt,
                    prev_messages=second_messages,
                ):
                    out_msg += chunk
                    print(chunk, end="", flush=True)
                content = out_msg
            else:
                if verbose_info:
                    print("[Agent] 第二轮调用（非流式）...")
                response2 = await self.llm_handler.fetch(
                    msg="请基于工具返回的结果继续回答。",
                    system_prompt=self.system_prompt,
                    prev_messages=second_messages,
                )
                content = response2.choices[0].message.content or ""

        else:
            # -------------------------------------------------
            # 4. 未请求 tool_calls：按 stream 参数决定输出方式
            # -------------------------------------------------
            if verbose_info:
                print("[Agent] 无 tool_calls")

            if stream:
                # 模拟流式输出：逐字打印到 stdout
                for char in content:
                    print(char, end="", flush=True)
                    # 极短延迟制造打字效果，不影响整体速度
                    await asyncio.sleep(0.0005)

        # -------------------------------------------------
        # 5. 将最终回复写入上下文
        # -------------------------------------------------
        await self.llm_context_hanlder.add_context(
            LLMContextPair(
                LLMContext(role="user", content=msg),
                LLMContext(role="assistant", content=content),
            )
        )

        return content

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
