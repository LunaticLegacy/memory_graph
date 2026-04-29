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
        hint = self.tool_registry.get_prompt_hint()
        if hint:
            prompt = f"{prompt}\n{hint}"
        return prompt

    async def round_call(
        self, 
        msg: str,
        verbose: bool = False
    ) -> str:
        """
        进行一整个轮次的 Agent 执行轮。

        Args:
            msg: 本 agent 的本次输入。
            verbose: 是否实施显示 agent 的输出。

        Returns:
            LLM 生成的完整回复文本。
        """
        out_msg = ""

        async for s in self.llm_handler.fetch_stream(
            msg=msg,
            system_prompt=self.system_prompt,
            prev_messages=await self.llm_context_hanlder.get_now_context(),
            max_tokens=9999999,
        ):
            if verbose:
                print(s, end="", flush=True)
            out_msg += s

        # 在生成完毕后，将内容加入到上下文。
        await self.llm_context_hanlder.add_context(
            LLMContextPair(
                LLMContext(role="user", content=msg),
                LLMContext(role="assistant", content=out_msg),
            )
        )

        # 提取所有的 JSON 信息，随后启用 tool call。
        json_msgs = self.extract_json_msg(out_msg)

        tool_results: List[Dict[str, Any]] = []
        for info in json_msgs:
            tr = await self.tool_call(info)
            if tr is not None:
                tool_results.append(tr)

        # 将所有工具输出统一写入上下文，供下一轮 LLM 使用。
        if tool_results:
            lines: List[str] = []
            for tr in tool_results:
                lines.append(f"[Tool Result: {tr['tool']}]\n{tr['result']}")
            combined = "\n\n".join(lines)
            await self.llm_context_hanlder.add_context(
                LLMContextPair(
                    LLMContext(role="user", content=combined),
                    LLMContext(role="assistant", content="Acknowledged."),
                )
            )

        return out_msg

    def extract_json_msg(self, msg: str) -> List[JSON]:
        """
        提取出一个 str 对象内所有的 JSON 对象。

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
        执行 tool call。
        期望的 JSON 格式: {"tool": "<name>", "arguments": {<key>: <value>, ...}}
        """
        if not isinstance(json_info, dict):
            return None

        tool_name = json_info.get("tool")
        if not tool_name or not isinstance(tool_name, str):
            return None

        arguments = json_info.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        # --- Execute tool ---
        try:
            result = await self.tool_registry.execute(tool_name, arguments)
        except Exception as exc:
            result = f"Error in executing tool: {exc}"

        return {"tool": tool_name, "result": result}
