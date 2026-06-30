from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, List, Optional

from .config import AgentConfig
from .llm.base import BaseLLMClient, LLMResponse
from .llm.openai_client import OpenAIClient
from .strategy import ExplorationPhase, KnowledgeGraph, StrategyEngine, StrategyState
from .tools.env_adapter import EnvironmentToolAdapter

logger = logging.getLogger(__name__)


class StrategicAgent:
    """Strategy-driven agent that uses programmatic exploration with LLM for reasoning.

    Unlike the pure ReAct agent, this agent uses the StrategyEngine to determine
    what actions to take (systematic combinatorial exploration, then analysis, then
    route optimization). The LLM is only consulted for:
    - Interpreting complex or ambiguous results
    - Planning when multiple viable routes exist
    - Deciding on optimization parameters
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[BaseLLMClient] = None,
        env_adapter: Optional[EnvironmentToolAdapter] = None,
    ):
        self.config = config or AgentConfig()
        self.llm = llm_client or OpenAIClient()
        self.env = env_adapter or EnvironmentToolAdapter()
        self.strategy: Optional[StrategyEngine] = None

        self._step_count = 0
        self._best_score: Optional[float] = None
        self._session_active = False
        self._task_constraints: Dict[str, Any] = {}

    def run(
        self,
        seed: Optional[int] = None,
        complexity_level: Optional[str] = None,
        task: Optional[Dict[str, Any]] = None,
        max_steps: Optional[int] = None,
        close_on_finish: bool = True,
    ) -> Dict[str, Any]:
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
        self._parse_task_constraints()

        self.strategy = StrategyEngine(constraints=self._task_constraints)

        if self.config.verbose:
            logger.info(f"Strategic session started. Max steps: {max_steps}")
            print(f"[Strategic Agent] Session started. Constraints: {json.dumps(self._task_constraints, default=str)}")

        try:
            while self._step_count < max_steps:
                done = self._strategic_step()
                if done:
                    break
                time_info = self._get_time_info()
                if time_info.get("time_remaining", 1) <= 0:
                    if self.config.verbose:
                        print("  [Done] Time budget exhausted.")
                    break
        except KeyboardInterrupt:
            logger.info("Session interrupted by user.")
        except Exception as e:
            logger.error(f"Session error at step {self._step_count}: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if close_on_finish:
                self.env.close()

        return {
            "best_score": self._best_score,
            "steps_taken": self._step_count,
            "session_id": self.env.session_id,
            "knowledge_summary": self.strategy.get_knowledge_summary() if self.strategy else "",
        }

    def _strategic_step(self) -> bool:
        self._step_count += 1

        time_info = self._get_time_info()
        elapsed = time_info.get("elapsed_time", self.strategy.state.time_elapsed)
        budget = time_info.get("time_budget", self.strategy.state.time_budget)
        self.strategy.set_time_info(elapsed, budget)

        actions = self.strategy.get_next_actions()

        if self.config.verbose:
            phase = self.strategy.state.phase
            print(f"\n{'─'*50}")
            print(f"  Step {self._step_count} | Phase: {phase}")
            print(f"{'─'*50}")

        for action_spec in actions:
            action = action_spec["action"]
            arguments = action_spec.get("arguments", {})

            if action == "finish_experiment":
                self._execute("finish_experiment", {})
                return True

            if self.config.verbose:
                args_str = json.dumps(arguments, ensure_ascii=False)
                print(f"  [Action] {action}({args_str})")

            result = self._execute(action, arguments)

            if self.config.verbose:
                result_str = json.dumps(result, ensure_ascii=False)
                print(f"  [Result] {result_str}")

            self.strategy.record_result(action, arguments, result)

            if action == "submit_solution" and isinstance(result, dict):
                if result.get("passed"):
                    cost = result.get("total_experiment_cost")
                    if cost is not None:
                        if self._best_score is None or cost < self._best_score:
                            self._best_score = cost
                            if self.config.verbose:
                                print(f"  [BEST] New best score: {cost}")

        if self.strategy.state.phase == ExplorationPhase.DONE:
            if not self.strategy.state.submitted:
                self._attempt_best_submission()
            self._execute("finish_experiment", {})
            return True

        return False

    def _attempt_best_submission(self) -> None:
        kg = self.strategy.state.kg
        qualifying = kg.get_qualifying_candidates()
        if qualifying:
            target = qualifying[0].name
        else:
            medicinal = kg.get_medicinal_candidates()
            if medicinal:
                target = medicinal[0].name
            else:
                return

        if self.config.verbose:
            print(f"  [Final Submit] Attempting submission with target: {target}")

        result = self._execute("submit_solution", {"target_compound": target})
        self.strategy.record_result("submit_solution", {"target_compound": target}, result)
        if isinstance(result, dict) and result.get("passed"):
            cost = result.get("total_experiment_cost")
            if cost is not None and (self._best_score is None or cost < self._best_score):
                self._best_score = cost

    def _execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.env.dispatch(name, arguments)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_task_constraints(self) -> None:
        try:
            desc_result = self.env.dispatch("task_description", {})
            if isinstance(desc_result, dict):
                task_desc = desc_result.get("task_description", desc_result)
                self._task_constraints = task_desc.get("constraints", {})
                time_budget = self._task_constraints.get("max_time_seconds", 14400)
                if self.strategy:
                    self.strategy.state.time_budget = float(time_budget)
        except Exception:
            self._task_constraints = {}

    def _get_time_info(self) -> Dict[str, Any]:
        try:
            backend = self.env._backend
            session = backend.get_session(self.env.session_id)
            return {
                "elapsed_time": session._elapsed_time,
                "time_budget": session._time_budget(),
                "time_remaining": session._time_remaining(),
                "total_experiment_cost": session._total_cost,
            }
        except Exception:
            return {"time_remaining": 999999}

    def _consult_llm(self, question: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": (
                "You are helping a chemistry exploration agent make decisions. "
                "The world has unique chemistry — real-world knowledge does NOT apply. "
                "Answer concisely based only on the experimental data provided."
            )},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}" if context else question},
        ]
        response = self.llm.chat(
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )
        return response.content
