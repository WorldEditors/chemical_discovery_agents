from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from xenoverse.sci_research_env.environment.backend import SciResearchBackend


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

    def create_session(self, **sampler_kwargs) -> Dict[str, Any]:
        result = self._backend.handle_request({
            "action": "sample_environment",
            "sampler_kwargs": sampler_kwargs,
        })
        if not result.get("success"):
            raise RuntimeError(f"Failed to create session: {result.get('message')}")
        self._session_id = result["session_id"]
        self._task_description = result.get("task_description", "")
        self._tool_prompt = result.get("tool_prompt", "")
        self._extract_tools()
        return result

    def load_session(self, task: Dict[str, Any]) -> Dict[str, Any]:
        result = self._backend.handle_request({
            "action": "create_session",
            "task": task,
        })
        if not result.get("success"):
            raise RuntimeError(f"Failed to load session: {result.get('message')}")
        self._session_id = result["session_id"]
        self._task_description = result.get("task_description", "")
        self._tool_prompt = result.get("tool_prompt", "")
        self._extract_tools()
        return result

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._session_id:
            raise RuntimeError("No active session. Call create_session() first.")
        result = self._backend.handle_request({
            "action": "dispatch_function_call",
            "session_id": self._session_id,
            "function_call": {"name": name, "arguments": arguments},
        })
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    def close(self) -> None:
        if self._session_id:
            self._backend.handle_request({
                "action": "close_session",
                "session_id": self._session_id,
            })
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
