from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .working import WorkingMemory
from .episodic import EpisodicMemory, Episode, ExperimentSummary
from .semantic import SemanticMemory


class MemoryManager:
    """Unified interface coordinating all three memory subsystems.

    Responsibilities:
    - Route new information to appropriate memory stores
    - Compose context from all stores for the LLM
    - Manage persistence (save/load)
    - Generate memory-augmented prompts
    """

    def __init__(
        self,
        working: Optional[WorkingMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        semantic: Optional[SemanticMemory] = None,
        memory_dir: Optional[str] = None,
    ):
        self.memory_dir = memory_dir
        if memory_dir:
            os.makedirs(memory_dir, exist_ok=True)

        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicMemory(
            storage_path=os.path.join(memory_dir, "episodic.json") if memory_dir else None
        )
        self.semantic = semantic or SemanticMemory(
            storage_path=os.path.join(memory_dir, "semantic.json") if memory_dir else None
        )

    def record_action(
        self,
        action: str,
        arguments: Dict[str, Any],
        observation: Dict[str, Any],
        reasoning: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        episode = Episode(
            action=action,
            arguments=arguments,
            observation=observation,
            reasoning=reasoning,
            tags=tags or [],
        )
        self.episodic.record(episode)
        self._extract_semantic(action, arguments, observation)

    def _extract_semantic(
        self, action: str, arguments: Dict[str, Any], observation: Dict[str, Any]
    ) -> None:
        if action == "analyze_compound":
            compound_id = arguments.get("compound_id", "")
            if isinstance(observation, dict) and observation.get("success", True):
                props = {k: v for k, v in observation.items() if k not in ("success", "error")}
                name = observation.get("name", compound_id)
                self.semantic.update_compound(compound_id, name=name, properties=props)

        elif action == "perform_reaction":
            if isinstance(observation, dict) and observation.get("success", True):
                rxn_id = observation.get("reaction_id", "")
                products = observation.get("products", [])
                reactants = list(arguments.get("reactant_amounts", {}).keys())
                conditions = {
                    "temperature_C": arguments.get("temperature_C"),
                    "pressure_atm": arguments.get("pressure_atm"),
                    "duration_seconds": arguments.get("duration_seconds"),
                }
                if rxn_id:
                    self.semantic.update_reaction(
                        rxn_id,
                        reactants=reactants,
                        products=[p if isinstance(p, str) else p.get("name", "") for p in products]
                        if isinstance(products, list) else [],
                        observed_conditions=conditions,
                    )

        elif action == "estimate_cost":
            if isinstance(observation, dict) and observation.get("success", True):
                cost_info = {
                    "conditions": arguments,
                    "estimated_cost": observation.get("estimated_cost"),
                }
                self.semantic.add_strategy(
                    insight=f"Cost estimate: {json.dumps(cost_info, ensure_ascii=False)[:200]}",
                    confidence=0.3,
                    tags=["cost_probe"],
                )

        elif action == "submit_solution":
            if isinstance(observation, dict):
                score = observation.get("aggregate_score")
                is_best = observation.get("is_new_best", False)
                if score is not None and is_best:
                    self.semantic.add_strategy(
                        insight=f"Best submission so far: score={score}, target={arguments.get('target_compound')}",
                        confidence=0.9,
                        tags=["submission", "best_score"],
                    )

    def compose_memory_context(self, max_chars: int = 4000) -> str:
        """Compose a memory summary suitable for injection into the system prompt."""
        parts = []

        semantic_summary = self.semantic.summarize()
        if semantic_summary.strip():
            parts.append(f"[Knowledge Base]\n{semantic_summary}")

        recent_episodes = self.episodic.get_recent(5)
        if recent_episodes:
            lines = ["[Recent Experiment History]"]
            for ep in recent_episodes:
                outcome = ep.outcome_quality or "unknown"
                lines.append(f"  - {ep.action}({json.dumps(ep.arguments, ensure_ascii=False)[:80]}) -> {outcome}")
            parts.append("\n".join(lines))

        past_summaries = self.episodic.get_summaries(3)
        if past_summaries:
            lines = ["[Past Session Insights]"]
            for s in past_summaries:
                lines.append(f"  - World {s.world_id}: best_score={s.best_score}, discoveries={s.key_discoveries[:3]}")
            parts.append("\n".join(lines))

        context = "\n\n".join(parts)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncated)"
        return context

    def create_session_summary(
        self,
        world_id: str,
        seed: int,
        best_score: Optional[float] = None,
        key_discoveries: Optional[List[str]] = None,
        successful_routes: Optional[List[Dict[str, Any]]] = None,
        failed_approaches: Optional[List[str]] = None,
        cost_insights: Optional[List[str]] = None,
    ) -> ExperimentSummary:
        summary = ExperimentSummary(
            world_id=world_id,
            seed=seed,
            total_steps=self.episodic.total_episodes,
            best_score=best_score,
            key_discoveries=key_discoveries or [],
            successful_routes=successful_routes or [],
            failed_approaches=failed_approaches or [],
            cost_insights=cost_insights or [],
        )
        self.episodic.record_summary(summary)
        return summary

    def save_all(self) -> None:
        self.episodic.save()
        self.semantic.save()

    def reset_working(self) -> None:
        self.working.clear()
