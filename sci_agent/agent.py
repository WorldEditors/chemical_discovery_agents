from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .llm.base import BaseLLMClient, LLMResponse
from .llm.openai_client import OpenAIClient
from .memory.manager import MemoryManager
from .memory.working import WorkingMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .tools.env_adapter import EnvironmentToolAdapter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
You are a scientific research agent exploring an unfamiliar chemistry world.
Real-world chemical knowledge does NOT apply — all compound names, reactions, and properties
are unique to this world. You must discover everything empirically through experimentation.

Your goal: {task_description}

Strategy guidelines:
1. Start by surveying available materials and tools.
2. Systematically explore reactions to discover new compounds.
3. Analyze compounds to assess medicinal potential and toxicity.
4. Use estimate_cost to understand the cost structure before committing to expensive experiments.
5. Track which approaches work and which fail.
6. When you have a promising route, submit it via submit_solution.
7. You may submit multiple times — refine your approach based on feedback.

{memory_context}

Think step-by-step. For each action, briefly explain your reasoning, then call the appropriate tool.
"""


class SciResearchAgent:
    """ReAct-style agent for scientific research exploration.

    Integrates LLM reasoning with environment interaction and long-term memory.
    Uses a think-act-observe loop where:
      - Think: LLM reasons about current state and decides next action
      - Act: Tool call dispatched to the environment
      - Observe: Result recorded in memory and fed back to LLM
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[BaseLLMClient] = None,
        env_adapter: Optional[EnvironmentToolAdapter] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.config = config or AgentConfig()
        self.llm = llm_client or OpenAIClient(
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self.env = env_adapter or EnvironmentToolAdapter()
        self.memory = memory_manager or MemoryManager(
            working=WorkingMemory(
                max_messages=self.config.working_memory_max_messages,
                max_tokens=self.config.working_memory_max_tokens,
            ),
            episodic=EpisodicMemory(
                storage_path=f"{self.config.memory_dir}/episodic.json"
                if self.config.memory_dir else None
            ),
            semantic=SemanticMemory(
                storage_path=f"{self.config.memory_dir}/semantic.json"
                if self.config.memory_dir else None
            ),
            memory_dir=self.config.memory_dir,
        )

        self._step_count = 0
        self._best_score: Optional[float] = None
        self._session_active = False

    def run(
        self,
        seed: Optional[int] = None,
        complexity_level: Optional[str] = None,
        task: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        close_on_finish: bool = True,
    ) -> Dict[str, Any]:
        """Run a full exploration session.

        Args:
            seed: Random seed for world generation.
            complexity_level: World complexity (easy/medium/hard).
            task: Pre-built task dict (skips sampling if provided).
            max_steps: Override max steps from config.
            close_on_finish: Whether to close the environment session when done.

        Returns:
            Session result dict with best_score, steps_taken, and summary.
        """
        max_steps = max_steps or self.config.max_steps
        self._step_count = 0
        self._best_score = None

        if task:
            session_info = self.env.load_session(task)
        else:
            sampler_kwargs = {}
            if seed is not None:
                sampler_kwargs["seed"] = seed
            elif self.config.seed is not None:
                sampler_kwargs["seed"] = self.config.seed
            if complexity_level:
                sampler_kwargs["complexity_level"] = complexity_level
            elif self.config.complexity_level:
                sampler_kwargs["complexity_level"] = self.config.complexity_level
            session_info = self.env.create_session(**sampler_kwargs)

        self._session_active = True
        task_description = self.env.get_task_description()

        memory_context = self.memory.compose_memory_context()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description,
            memory_context=f"\n[Prior Experience]\n{memory_context}" if memory_context else "",
        )
        self.memory.working.set_system(system_prompt)

        observation_text = json.dumps(session_info.get("observation", {}), ensure_ascii=False)
        self.memory.working.add_message("user", f"Session started. Initial observation:\n{observation_text}")

        tools_schema = self.env.get_openai_tools_schema()

        if self.config.verbose:
            logger.info(f"Session started. Max steps: {max_steps}")

        try:
            while self._step_count < max_steps:
                step_result = self._step(tools_schema)
                if step_result.get("done"):
                    break
        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        except Exception as e:
            logger.error(f"Session error at step {self._step_count}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            self._finalize_session(close_env=close_on_finish)

        return {
            "best_score": self._best_score,
            "steps_taken": self._step_count,
            "session_id": self.env.session_id,
            "memory_summary": self.memory.semantic.summarize(),
        }

    def _step(self, tools_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._step_count += 1

        if self.config.verbose:
            logger.info(f"Step {self._step_count}")

        messages = self.memory.working.get_messages()
        response = self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if response.has_tool_calls:
            self.memory.working.add_tool_call(response.content, response.tool_calls)

            for tc in response.tool_calls:
                result = self._execute_tool(tc["name"], tc["arguments"])
                result_str = json.dumps(result, ensure_ascii=False)
                self.memory.working.add_tool_result(tc["id"], tc["name"], result_str)

                self.memory.record_action(
                    action=tc["name"],
                    arguments=tc["arguments"],
                    observation=result,
                    reasoning=response.content or "",
                    tags=self._infer_tags(tc["name"], tc["arguments"], result),
                )

                if tc["name"] == "submit_solution" and isinstance(result, dict):
                    score = result.get("aggregate_score")
                    if score is not None:
                        if self._best_score is None or score > self._best_score:
                            self._best_score = score

                if self.config.verbose:
                    logger.info(f"  Tool: {tc['name']} -> {result_str[:200]}")

            return {"done": False}
        else:
            self.memory.working.add_message("assistant", response.content)
            if self._is_done_signal(response.content):
                return {"done": True}
            self.memory.working.add_message(
                "user",
                "Continue your exploration. Use tools to gather more information or submit a solution."
            )
            return {"done": False}

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.env.dispatch(name, arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_done_signal(self, content: str) -> bool:
        if not content:
            return False
        done_indicators = [
            "I have completed",
            "exploration is complete",
            "no further improvements",
            "final submission",
            "I am done",
            "task complete",
        ]
        content_lower = content.lower()
        return any(ind.lower() in content_lower for ind in done_indicators)

    def _infer_tags(self, action: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        tags = [action]
        if action == "perform_reaction":
            tags.append("experiment")
        elif action == "analyze_compound":
            tags.append("analysis")
        elif action == "submit_solution":
            tags.append("submission")
            if isinstance(result, dict) and result.get("is_new_best"):
                tags.append("best_score")
        elif action == "purchase":
            tags.append("acquisition")
        elif action == "estimate_cost":
            tags.append("cost_probe")
        return tags

    def _finalize_session(self, close_env: bool = True) -> None:
        if not self._session_active:
            return
        self._session_active = False

        key_discoveries = []
        for ck in self.memory.semantic.get_compounds_with_medicinal_hints():
            key_discoveries.append(f"Medicinal compound: {ck.name}")

        strategies = self.memory.semantic.get_strategies(min_confidence=0.5)
        cost_insights = [s.insight for s in strategies if "cost" in s.insight.lower()]

        self.memory.create_session_summary(
            world_id=self.env.session_id or "unknown",
            seed=self.config.seed or 0,
            best_score=self._best_score,
            key_discoveries=key_discoveries[:10],
            cost_insights=cost_insights[:5],
        )

        self.memory.save_all()

        if close_env:
            self.env.close()

        if self.config.verbose:
            logger.info(
                f"Session finalized. Steps: {self._step_count}, Best score: {self._best_score}"
            )
