from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class WorkingMemory:
    """Short-term memory that holds the current conversation context.

    Manages the message history within the LLM context window, handling
    truncation when the window is exceeded.
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
        import json as _json
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
        result.extend(self.messages)
        return result

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def __len__(self) -> int:
        return len(self.messages)
