from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

_xenoverse_root = os.environ.get(
    "XENOVERSE_ROOT",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "Xenoverse"),
)
if os.path.isdir(_xenoverse_root) and _xenoverse_root not in sys.path:
    sys.path.insert(0, os.path.abspath(_xenoverse_root))

from xenoverse.chemverse.environment.backend import SciResearchBackend


class EnvironmentToolAdapter:
    """Adapter that connects the agent to the sci_research environment.

    Wraps the SciResearchBackend to provide:
    - Session lifecycle (create, dispatch, close)
    - Tool schema extraction for LLM function calling
    - Result parsing and error handling
    """

    def __init__(self, backend: Optional[SciResearchBackend] = None):
        self._backend = backend or SciResearchBackend()
        self._session_id: Optional[str] = None
        self._task_description: str = ""
        self._tool_prompt: str = ""
        self._function_tools: List[Dict[str, Any]] = []
        self.trace_sink: Optional[Callable[..., None]] = None
        self.trace_step = 0
        self.hide_world_id = False

    def _agent_view(self, value: Any) -> Any:
        """Remove hidden world identity from agent-visible environment data."""
        if not self.hide_world_id:
            return value
        if isinstance(value, dict):
            return {key: self._agent_view(item) for key, item in value.items()
                    if key != "world_id"}
        if isinstance(value, list):
            return [self._agent_view(item) for item in value]
        return value

    def _emit(self, event: str, input: Any, output: Any, session_id: Optional[str] = None) -> None:
        if self.trace_sink:
            self.trace_sink(
                event=event,
                step=self.trace_step,
                session_id=session_id if session_id is not None else self._session_id,
                input=input,
                output=output,
            )

    @staticmethod
    def _public_session_result(result: Dict[str, Any]) -> Dict[str, Any]:
        return {key: result[key] for key in (
            "success", "session_id", "task_description", "tool_prompt", "observation", "message"
        ) if key in result}

    def create_session(self, **sampler_kwargs) -> Dict[str, Any]:
        result = self._backend.handle_request({
            "action": "sample_environment",
            "sampler_kwargs": sampler_kwargs,
        })
        result = self._agent_view(result)
        if not result.get("success"):
            self._emit("session_start", {"mode": "sampled", "sampler_kwargs": sampler_kwargs},
                       self._public_session_result(result))
            raise RuntimeError(f"Failed to create session: {result.get('message')}")
        self._session_id = result["session_id"]
        self._task_description = result.get("task_description", "")
        self._tool_prompt = result.get("tool_prompt", "")
        self._extract_tools()
        self._emit(
            "session_start",
            {"mode": "sampled", "sampler_kwargs": sampler_kwargs},
            {**self._public_session_result(result), "tools": self._function_tools},
        )
        return result

    def load_session(self, task: Dict[str, Any]) -> Dict[str, Any]:
        result = self._backend.handle_request({
            "action": "create_session",
            "task": task,
        })
        result = self._agent_view(result)
        if not result.get("success"):
            self._emit("session_start", {"mode": "prebuilt"}, self._public_session_result(result))
            raise RuntimeError(f"Failed to load session: {result.get('message')}")
        self._session_id = result["session_id"]
        self._task_description = result.get("task_description", "")
        self._tool_prompt = result.get("tool_prompt", "")
        self._extract_tools()
        # The prebuilt task contains hidden world ground truth. Only the public
        # session response belongs in an agent/environment trajectory.
        self._emit(
            "session_start",
            {"mode": "prebuilt"},
            {**self._public_session_result(result), "tools": self._function_tools},
        )
        return result

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        tool_input = {"name": name, "arguments": arguments}
        try:
            result = self._backend.handle_request({
                "action": "dispatch_function_call",
                "session_id": self._session_id,
                "function_call": tool_input,
            })
            output = result["result"] if isinstance(result, dict) and "result" in result else result
            output = self._agent_view(output)
            self._emit("tool_call", tool_input, output)
            return output
        except Exception as exc:
            self._emit("tool_call", tool_input, {"error": str(exc)})
            raise

    def close(self) -> None:
        if self._session_id:
            session_id = self._session_id
            result = self._backend.handle_request({
                "action": "close_session",
                "session_id": session_id,
            })
            self._emit("session_close", {}, result, session_id=session_id)
            self._session_id = None

    def get_openai_tools_schema(self) -> List[Dict[str, Any]]:
        return self._function_tools

    def get_task_description(self) -> str:
        return self._task_description

    def get_tool_prompt(self) -> str:
        return self._tool_prompt

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def _extract_tools(self) -> None:
        if not self._session_id:
            return
        summary = self._backend.handle_request({
            "action": "get_session_summary",
            "session_id": self._session_id,
        })
        raw_tools = summary.get("function_tools", [])
        self._function_tools = []
        for tool in raw_tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                self._function_tools.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", fn.get("brief", "")),
                        "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                    },
                })
