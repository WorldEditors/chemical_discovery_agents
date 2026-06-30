from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkingMemory:
    """Short-term memory that holds the current conversation context.

    Manages the message history within the LLM context window, handling
    truncation when the window is exceeded. Trimming is structure-aware:
    it never leaves orphaned tool-result messages without their preceding
    assistant message containing tool_calls.
    """

    max_messages: int = 100
    max_tokens: int = 32000
    messages: List[Dict[str, Any]] = field(default_factory=list)
    _system_message: Dict[str, Any] = field(default=None)

    def set_system(self, content: str) -> None:
        self._system_message = {"role": "system", "content": content}

    def add_message(self, role: str, content: str, **kwargs) -> None:
        msg: Dict[str, Any] = {"role": role, "content": content}
        msg.update(kwargs)
        self.messages.append(msg)
        self._trim()

    def add_tool_call(self, assistant_content: str, tool_calls: List[Dict[str, Any]]) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content or ""}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": _json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ]
        self.messages.append(msg)
        self._trim()

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        })
        self._trim()

    def get_messages(self) -> List[Dict[str, Any]]:
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._sanitized(self.messages))
        return result

    @staticmethod
    def _sanitized(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a structurally valid copy of the message list.

        Guarantees the constraints the chat API enforces, which can otherwise
        be broken by context-window trimming:
        - every ``tool`` message has a matching preceding assistant ``tool_calls``
          (orphaned tool results are dropped);
        - every assistant ``tool_calls`` entry is answered by a following
          ``tool`` message (unanswered calls are stripped, since the partner
          tool results were trimmed away);
        - empty assistant messages (no content and no tool_calls) are dropped.
        """
        result: List[Dict[str, Any]] = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            role = msg.get("role")

            if role == "tool":
                i += 1
                continue

            if role == "assistant" and msg.get("tool_calls"):
                answered: Dict[str, Dict[str, Any]] = {}
                j = i + 1
                while j < n and messages[j].get("role") == "tool":
                    tcid = messages[j].get("tool_call_id")
                    if tcid is not None:
                        answered[tcid] = messages[j]
                    j += 1

                kept_calls = [
                    tc for tc in msg["tool_calls"] if tc.get("id") in answered
                ]

                if kept_calls:
                    new_msg = dict(msg)
                    new_msg["tool_calls"] = kept_calls
                    result.append(new_msg)
                    for tc in kept_calls:
                        result.append(answered[tc["id"]])
                elif (msg.get("content") or "").strip():
                    new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    result.append(new_msg)

                i = j
                continue

            if role == "assistant" and not (msg.get("content") or "").strip():
                i += 1
                continue

            result.append(msg)
            i += 1

        return result

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        while len(self.messages) > self.max_messages and self.messages:
            self._remove_oldest_turn()
        while len(self.messages) > 1 and self._current_tokens() > self.max_tokens:
            self._remove_oldest_turn()

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """Rough token estimate (~4 chars/token) over the serialized messages.

        Mirrors the fallback heuristic in the LLM client's ``count_tokens`` so
        trimming can bound the prompt size without a tokenizer dependency.
        """
        total = 0
        for m in messages:
            total += len(_json.dumps(m, ensure_ascii=False)) // 4 + 4
        return total

    def _current_tokens(self) -> int:
        msgs = ([self._system_message] if self._system_message else []) + self.messages
        return self._estimate_tokens(msgs)

    def _remove_oldest_turn(self) -> None:
        """Remove the oldest complete turn (assistant+tool_results or single message).

        Ensures we never leave orphaned tool messages without their
        preceding assistant message with tool_calls.
        """
        if not self.messages:
            return

        first = self.messages[0]

        if first.get("role") == "assistant" and "tool_calls" in first:
            tool_call_ids = {tc["id"] for tc in first.get("tool_calls", [])}
            self.messages.pop(0)
            while self.messages and self.messages[0].get("role") == "tool":
                if self.messages[0].get("tool_call_id") in tool_call_ids:
                    self.messages.pop(0)
                else:
                    break
        elif first.get("role") == "tool":
            while self.messages and self.messages[0].get("role") == "tool":
                self.messages.pop(0)
        else:
            self.messages.pop(0)

    def __len__(self) -> int:
        return len(self.messages)
