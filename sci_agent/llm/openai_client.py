from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

_NON_RETRYABLE_ERROR_CODES = {"1010"}


def _describe_messages(messages: List[Dict[str, Any]]) -> str:
    """Compact structural summary of a messages array for diagnostics.

    Shows the role sequence plus tool-call/tool-result ids so malformed
    sequences (orphaned tool results or unanswered tool_calls) are visible
    without dumping full message contents.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            parts.append(f"assistant(tool_calls={ids})")
        elif role == "tool":
            parts.append(f"tool(reply_to={m.get('tool_call_id')})")
        else:
            empty = "" if (m.get("content") or "").strip() else " EMPTY"
            parts.append(f"{role}{empty}")
    return f"messages[{len(messages)}] structure: " + " | ".join(parts)


DEFAULT_LLM_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "llm.json",
)


def _load_llm_config(path: str) -> Dict[str, Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"LLM config not found at {path}; using empty config.")
        return {}


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        config_path: str = DEFAULT_LLM_CONFIG_PATH,
    ):
        cfg = _load_llm_config(config_path)

        self.model = model or cfg.get("model")
        self._timeout = timeout if timeout is not None else cfg.get("timeout", 120.0)
        self._max_retries = (
            max_retries if max_retries is not None else cfg.get("max_retries", 3)
        )

        self._client = OpenAI(
            api_key=api_key or cfg.get("api_key"),
            base_url=base_url or cfg.get("base_url"),
            timeout=self._timeout,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float=1.0,
        max_tokens: int=4096,
        **kwargs,
    ) -> LLMResponse:
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        last_error = None
        response = None
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(**call_kwargs)
                if response and response.choices:
                    break

                api_error = getattr(response, "error", None)
                if api_error is not None:
                    code = str(api_error.get("code")) if isinstance(api_error, dict) else None
                    detail = api_error.get("message") if isinstance(api_error, dict) else str(api_error)
                    if code in _NON_RETRYABLE_ERROR_CODES:
                        raise RuntimeError(
                            f"LLM rejected the request (code {code}): {detail}. "
                            f"This is a request-validation error and will not be retried. "
                            f"{_describe_messages(messages)}"
                        )
                    last_error = RuntimeError(f"LLM API error (code {code}): {detail}")
                else:
                    last_error = RuntimeError(f"LLM returned empty response: {response}")

                wait = 10 + 5 * attempt
                logger.warning(f"{last_error} (attempt {attempt+1}/{self._max_retries}). Retrying in {wait}s...")
                time.sleep(wait)
            except RuntimeError:
                raise
            except Exception as e:
                last_error = e
                wait = 10 + 5 * attempt
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self._max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"LLM call failed after {self._max_retries} retries: {last_error}")

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
