from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseLLMClient, LLMResponse


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        final_key = (api_key or "").strip()
        final_url = (base_url or "").strip() or None

        if not final_key:
            raise ValueError("Missing LLM api_key. Set it in configs/default.json or pass a config file.")

        self._client = OpenAI(
            api_key=final_key,
            base_url=final_url,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**call_kwargs)

        if not response or not response.choices:
            raise RuntimeError(f"LLM returned empty response: {response}")

        choice = response.choices[0]
        message = choice.message

        if message is None:
            raise RuntimeError(f"LLM returned None message in choice: {choice}")

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            raw=response,
        )

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            total = 0
            for msg in messages:
                total += 4
                for key, val in msg.items():
                    if isinstance(val, str):
                        total += len(enc.encode(val))
            total += 2
            return total
        except Exception:
            return sum(len(json.dumps(m)) // 4 for m in messages)
