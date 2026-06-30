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
You are a scientific research agent exploring a completely unfamiliar chemistry world.
Real-world chemical knowledge does NOT apply — all compound names, reactions, and properties
are unique to this world. You must discover everything empirically through experimentation.

CRITICAL EPISTEMIC PRINCIPLE:
You have ZERO prior knowledge about this world. Do NOT assume anything about:
- How many reactants a reaction requires (could be 1, 2, 3, 4, or more)
- Which chemicals are "reactive" vs "inert" — you cannot tell until you try
- What temperature or pressure is needed — vary systematically
- What role a chemical plays — solvents might be reactants, expensive chemicals might be catalysts
The ONLY way to learn is to observe. If a combination produces no result, that is ONE data point —
it does NOT tell you anything about other combinations involving those same chemicals.

Your goal: {task_description}

{tool_prompt}

=== WORLD MECHANICS ===

SOLVENTS & DISSOLUTION:
- Some purchasable chemicals are solvents (role: "solvent"). They are very cheap.
- Reactions involving solid or gas reactants REQUIRE a solvent that dissolves them to proceed.
- If a reactant is itself a liquid solvent, it can serve as the reaction medium.
- Otherwise, add an external solvent to dissolve solid/high-mp reactants.
- More solvent volume → better dissolution → faster reaction rate.
- Without adequate dissolution, reactions are blocked entirely (no reaction occurs).
- Reaction results include "observations" describing what dissolved, what remained solid, etc.

EQUIPMENT & CAPACITY:
- Each equipment type has a max_capacity_g (total mass limit for all materials).
- Exceeding capacity is rejected instantly with no time or cost penalty.
- Equipment also has temperature and pressure limits.
- Use list_equipment to check capacity and limits before large-scale reactions.
- Larger equipment (autoclave: 2000g, reflux: 1000g) costs more per hour.

PRE-CHECK RULES (instant rejection, NO time or cost penalty):
- Insufficient inventory: you don't have enough of a chemical.
- Total reactant mass below 1g: too little material to perform or observe a reaction.
- Total mass exceeds equipment capacity: use smaller amounts or bigger equipment.
- Temperature/pressure setting exceeds equipment limits: adjust settings or use different equipment.
These are checked BEFORE the reaction starts. No materials are consumed, no time passes.

MID-REACTION EXPLOSION (causes material loss):
- If temperature or pressure rises DURING the reaction (e.g. from exothermic heat) and
  exceeds equipment limits, the equipment fails and ALL materials are destroyed.
- This is different from the pre-check: you set safe initial conditions, but the reaction
  itself generates enough heat/pressure to exceed limits. Use sealed/rated equipment for
  vigorous reactions or reduce reactant amounts.

OBSERVATIONS & PHENOMENA:
- Reaction results include an "observations" field with detailed physical phenomena:
  phase transitions (boiling, melting, solidification), dissolution behavior,
  gas escape, temperature changes, pressure buildup, and conversion progress.
- Use these observations to understand what is happening and adjust conditions.
- Example: "XXX went into vigorous ebullition" means the compound boiled off —
  lower the temperature or use sealed equipment.

COSTS:
- Each compound analysis costs 5 credits + 300s of time.
- Failed reactions (no applicable reaction found) cost 3 credits cleanup + full duration.
- Purification is expensive and scales superlinearly with mixture complexity.
- Solvents are very cheap (< 0.05 credits/g) — use them freely.
- Score = total experiment cost. Lower is better.

=== CRITICAL COST RULES ===
- Your score = TOTAL experiment cost. Every purchase and reaction counts. LOWER IS BETTER.
- NEVER buy large amounts (>10g) of expensive chemicals during exploration.
- Exploration phase: buy only 1-2g of each non-solvent chemical. Solvents are cheap — buy 10-15g.
- Production phase (after finding a working route): scale up ONLY the minimum needed for yield.
- The optimal cost for most tasks is 100-1000 credits. If you've spent >2000 credits without
  a passing submission, you are likely on the wrong track. Reconsider your approach.
- ALWAYS check price_per_gram before purchasing. Cheap chemicals (solvents) < 0.1 credits/g.
  Expensive chemicals > 3 credits/g — use sparingly.

=== STRATEGY GUIDELINES ===

PHASE 1 - SURVEY (Steps 1-3):
1. Call list_purchasable to see all available chemicals and their prices.
2. Identify solvents (very cheap, liquid) vs expensive reagents.
3. Call list_equipment to understand capacity/temperature limits.

PHASE 2 - SMALL-SCALE EXPLORATION (Steps 4-25):
4. Buy small amounts (1-2g) of EVERY non-solvent chemical + 10-15g of each solvent.
   Every purchasable chemical exists for a reason. Do not exclude any from exploration.
5. Systematically explore combinations of varying size:
   - Try pairs (A+B), triples (A+B+C), quadruples (A+B+C+D), etc.
   - Always include a solvent for dissolution unless all reactants are already liquid.
   - You have NO basis to assume how many components a reaction needs. Try all subset sizes.
   - If all pairs fail, this does NOT mean the chemicals are inert — try larger subsets.
6. Vary conditions systematically:
   - Temperature: try 100°C, then 150°C, then 200°C. Do not fixate on one temperature.
   - Duration: most reactions need 300-1200 seconds. Do not use < 60s.
   - Equipment: use appropriate equipment for your temperature/pressure needs.
7. CRITICAL: Read "observations" carefully:
   - "dissolving completely" = good, reaction can proceed
   - "only slightly dissolving" or "settled at the bottom" = poor dissolution, try different solvent
   - "conversion 0%" with short duration = need LONGER duration
   - "ebullition/evaporated" = temperature too high or need sealed equipment
8. When you get products (num_products_formed > 0), call get_inventory then analyze_compound.
9. Analyze products to check medicinal value and toxicity.
10. When a combination works, do ABLATION: remove one component at a time to find the
    minimal required set. This saves cost in production phase.
11. When a reaction succeeds but with low conversion or yield, investigate the cause:
    try longer duration, different temperatures, better dissolution (more solvent or a
    different solvent), or sealed equipment to prevent gas escape. Read the observations
    carefully — they tell you exactly what went wrong.
12. If the product doesn't meet requirements, it might be an INTERMEDIATE — try using it
    as a reactant in further reactions with other chemicals.

PHASE 3 - ROUTE OPTIMIZATION (Steps 25-40):
12. Once you find a compound meeting requirements, optimize the route.
13. Test different temperatures with small amounts to find the sweet spot.
14. Higher temperature generally = faster reaction but higher energy cost and risk of boil-off.
15. Find the minimum temperature that gives >80% conversion in reasonable time.
16. Use estimate_cost to compare different conditions.

PHASE 4 - PRODUCTION (Steps 40-50):
17. Scale up ONLY when you have a confirmed working route to a qualifying compound.
18. Calculate minimum input needed for required yield. Don't over-purchase.
19. Submit via submit_solution.
20. If submission fails (yield too low), scale up slightly and retry.

KEY MISTAKES TO AVOID:
- DO NOT buy hundreds of grams during exploration — this wastes money.
- DO NOT run reactions shorter than 60 seconds — most reactions need 300-1200 seconds.
- DO NOT keep repeating the same chemicals if dissolution is poor — try different solvents.
- DO NOT fixate on one route — if after 3 attempts it doesn't work, try other chemicals.
- DO NOT ignore dissolution observations — if material "settled at the bottom", the reaction
  CANNOT proceed efficiently. You MUST find a solvent that dissolves your reactants.
- DO NOT assume that a failed pair means those chemicals are useless — they may require
  additional components to react. Absence of evidence is not evidence of absence.
- DO NOT skip chemicals because they seem expensive — buy small amounts of everything.
- DO NOT stop exploring after finding one product — it may not meet requirements, and you
  may need to use it as input for a subsequent reaction to reach the final target.
- It is POSSIBLE that no combination of available chemicals can produce a compound meeting
  all task requirements. If after thorough and systematic exploration (trying many combinations
  of varying sizes, temperatures, and conditions) you are confident that no solution exists,
  call finish_experiment with no_solution=true. Only declare this after exhaustive exploration —
  an incorrect declaration scores 0.

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
        self.llm = llm_client or OpenAIClient()
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
        self._consecutive_no_tool = 0

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
        self._max_steps = max_steps
        task_description = self.env.get_task_description()
        tool_prompt = self.env.get_tool_prompt()

        memory_context = self.memory.compose_memory_context()
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            task_description=task_description,
            tool_prompt=f"\n[Tool Usage Guide]\n{tool_prompt}" if tool_prompt else "",
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
            print(f"\n{'─'*50}")
            print(f"  Step {self._step_count}/{self._max_steps}")
            print(f"{'─'*50}")

        self._inject_progress_context()

        messages = self.memory.working.get_messages()
        response = self.llm.chat(
            messages=messages,
            tools=tools_schema,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if response.has_tool_calls:
            self._consecutive_no_tool = 0
            if self.config.verbose and response.content:
                print(f"  [Reasoning] {response.content}")

            self.memory.working.add_tool_call(response.content, response.tool_calls)

            for tc in response.tool_calls:
                if self.config.verbose:
                    args_str = json.dumps(tc["arguments"], ensure_ascii=False)
                    print(f"  [Action] {tc['name']}({args_str})")

                result = self._execute_tool(tc["name"], tc["arguments"])
                if (isinstance(result, dict)
                    and not result.get("success", True)
                    and "Insufficient" in result.get("message", "")):
                    result["hint"] = (
                        "You don't have enough of this chemical. Either purchase more first, "
                        "or reduce the amount in your reaction."
                    )
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
                    if result.get("passed"):
                        cost = result.get("total_experiment_cost")
                        if cost is not None:
                            if self._best_score is None or cost < self._best_score:
                                self._best_score = cost

                if tc["name"] == "finish_experiment":
                    if self.config.verbose:
                        print(f"  [Observe] {result_str}")
                        print(f"  [Done] Agent called finish_experiment.")
                    return {"done": True}

                if self.config.verbose:
                    print(f"  [Observe] {result_str}")

            return {"done": False}
        else:
            self._consecutive_no_tool += 1
            if self.config.verbose and response.content:
                print(f"  [Thought] {response.content}")

            self.memory.working.add_message("assistant", response.content)
            if self._is_done_signal(response.content):
                if self.config.verbose:
                    print(f"  [Done] Agent signaled completion.")
                return {"done": True}

            if self._consecutive_no_tool >= 3:
                self.memory.working.add_message(
                    "user",
                    "You have not used any tools for several turns. You MUST call a tool now. "
                    "Either purchase chemicals, perform a reaction, analyze a compound, or call "
                    "finish_experiment if you cannot proceed. Check your current inventory above "
                    "and decide on a concrete action."
                )
            else:
                self.memory.working.add_message(
                    "user",
                    "Continue your exploration. Use tools to gather more information or submit a solution."
                )
            return {"done": False}

    def _inject_progress_context(self) -> None:
        """Update the system prompt with current progress, inventory, and discovered knowledge."""
        time_info = self._get_time_info()
        elapsed = time_info.get("elapsed_time", 0)
        budget = time_info.get("time_budget", 0)
        remaining = time_info.get("time_remaining", 0)
        total_cost = time_info.get("total_experiment_cost", 0)

        progress_lines = [
            f"[Session Progress] Step {self._step_count}/{self._max_steps}",
            f"  Time: {elapsed:.0f}s / {budget:.0f}s ({remaining:.0f}s remaining)",
            f"  Total experiment cost so far: {total_cost:.2f} credits",
            f"  Best submission: {self._best_score if self._best_score is not None else 'no passing submission yet'} (lower is better)",
        ]

        # Current inventory from environment
        try:
            inventory = self._execute_tool("get_inventory", {})
            if isinstance(inventory, dict):
                inv_entries = []
                for name, val in inventory.items():
                    if isinstance(val, dict):
                        amt = val.get("amount_g", 0)
                        if amt > 0:
                            inv_entries.append(f"{name} ({amt:.2f}g)")
                    elif isinstance(val, (int, float)) and val > 0:
                        inv_entries.append(f"{name} ({val:.2f}g)")
                if inv_entries:
                    progress_lines.append(f"  Current inventory: {', '.join(inv_entries)}")
                else:
                    progress_lines.append("  Current inventory: empty")
        except Exception:
            pass

        # Discovered compounds from semantic memory
        compounds = self.memory.semantic.get_all_compounds()
        if compounds:
            progress_lines.append(f"  Discovered compounds ({len(compounds)}):")
            for ck in compounds[:15]:
                props = ck.properties
                desc_parts = [ck.name]
                if "biological_activity" in props:
                    desc_parts.append(f"bio={props['biological_activity']}")
                if "toxicity_level" in props:
                    desc_parts.append(f"tox={props['toxicity_level']}")
                if "state_at_room_temp" in props:
                    desc_parts.append(f"state={props['state_at_room_temp']}")
                progress_lines.append(f"    - {', '.join(desc_parts)}")
            if len(compounds) > 15:
                progress_lines.append(f"    ... and {len(compounds) - 15} more")

        # Discovered reactions from semantic memory
        reactions = self.memory.semantic.get_all_reactions()
        if reactions:
            progress_lines.append(f"  Discovered reactions ({len(reactions)}):")
            for rk in reactions[:10]:
                reactants_str = " + ".join(rk.reactants) if rk.reactants else "?"
                products_str = " + ".join(rk.products) if rk.products else "?"
                cond_strs = []
                for cond in rk.observed_conditions[-2:]:
                    if isinstance(cond, dict):
                        parts = []
                        if cond.get("temperature_C") is not None:
                            parts.append(f"T={cond['temperature_C']}°C")
                        if cond.get("pressure_atm") is not None:
                            parts.append(f"P={cond['pressure_atm']}atm")
                        if cond.get("duration_seconds") is not None:
                            parts.append(f"t={cond['duration_seconds']}s")
                        if parts:
                            cond_strs.append(f"[{', '.join(parts)}]")
                conditions_info = f" (tried: {' '.join(cond_strs)})" if cond_strs else ""
                progress_lines.append(f"    - {reactants_str} → {products_str}{conditions_info}")
            if len(reactions) > 10:
                progress_lines.append(f"    ... and {len(reactions) - 10} more")

        # Medicinal candidates
        medicinal = self.memory.semantic.get_compounds_with_medicinal_hints()
        if medicinal:
            progress_lines.append(f"  Medicinal candidates: {[c.name for c in medicinal[:5]]}")

        # Strategic insights
        strategies = self.memory.semantic.get_strategies(min_confidence=0.7)
        if strategies:
            progress_lines.append("  Key insights:")
            for s in strategies[-3:]:
                progress_lines.append(f"    - {s.insight}")

        progress_text = "\n".join(progress_lines)

        current_system = self.memory.working._system_message
        if current_system:
            base = current_system["content"]
            marker = "\n\n[Session Progress]"
            if marker in base:
                base = base[:base.index(marker)]
            self.memory.working._system_message = {
                "role": "system",
                "content": base + "\n\n" + progress_text,
            }

    def _get_time_info(self) -> Dict[str, Any]:
        try:
            state = self.env.dispatch("task_description", {})
            if isinstance(state, dict):
                task_desc = state.get("task_description", {})
                constraints = task_desc.get("constraints", {})
                budget = constraints.get("max_time_seconds", 28800)
            else:
                budget = 28800
        except Exception:
            budget = 28800

        try:
            from xenoverse.chemverse.environment.backend import SciResearchBackend
            backend = self.env._backend
            session = backend.get_session(self.env.session_id)
            return {
                "elapsed_time": session._elapsed_time,
                "time_budget": session._time_budget(),
                "time_remaining": session._time_remaining(),
                "total_experiment_cost": session._total_cost,
            }
        except Exception:
            return {
                "elapsed_time": 0,
                "time_budget": budget,
                "time_remaining": budget,
                "total_experiment_cost": 0,
            }

    def _execute_tool(self, name: str, arguments: Dict[str, Any], record: bool = True) -> Dict[str, Any]:
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
