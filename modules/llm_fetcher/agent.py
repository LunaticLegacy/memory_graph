import json
from typing import Dict, List, Union

from .llm_fetcher import LLMFetcher
from .llm_context import (LLMContextPair, LLMContext, LLMContextHandler)

JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]

class Agent:

    def __init__(
        self,
        llm_handler: LLMFetcher,
        system_prompt: str
    ):
        self.llm_handler: LLMFetcher = llm_handler

        # 系统提示词
        self.system_prompt = system_prompt

        # 记忆，用于从上下文管理器内单独提取。
        self.memory_list: List[str] = []

        # 管理上下文用的
        self.llm_context_hanlder = LLMContextHandler(llm_handler=self.llm_handler)

        # 管理工具——工具本身要怎么设计？

    async def round_call(self, msg: str):
        """
        进行一整个轮次的 Agent 执行轮。

        Args:
            msg: 本 agent 的本次输入。
        """

        out_msg: str = ""
        
        async for s in self.llm_handler.fetch_stream(
            msg=msg,
            system_prompt=self.system_prompt, 
            prev_messages=await self.llm_context_hanlder.get_now_context(),
            max_tokens=16384
        ):
            out_msg += s

        # 在生成完毕后，将内容加入到上下文。   
        self.llm_context_hanlder.add_context(
            LLMContextPair(msg, out_msg)
        )

        # 提取所有的 JSON 信息，随后启用 tool call - 需要事先约定执行 tool call 的方法。
        json_msg: List[JSON] = self.extract_json_msg(out_msg)

        for info in json_msg:
            self.tool_call(info)


    async def extract_json_msg(self, msg: str) -> List[JSON]:
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

    async def tool_call(self, json_info: JSON):
        """
        执行 tool call。
        注意：这是一个 async 的过程，工具本身也可以被挂起。

        """
        pass