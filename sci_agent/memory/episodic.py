from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Episode:
    """A single experiment episode recording action, observation, and outcome."""

    action: str
    arguments: Dict[str, Any]
    observation: Dict[str, Any]
    reasoning: str = ""
    outcome_quality: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Episode:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentSummary:
    """Summary of a full experiment session."""

    world_id: str
    seed: int
    total_steps: int
    best_score: Optional[float]
    key_discoveries: List[str]
    successful_routes: List[Dict[str, Any]]
    failed_approaches: List[str]
    cost_insights: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentSummary:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class EpisodicMemory:
    """Long-term storage of past experiment episodes and session summaries.

    Supports persistence to disk so experience accumulates across runs.
    Provides retrieval by recency, relevance (tag matching), and outcome quality.
    """

    def __init__(self, storage_path: Optional[str] = None, max_episodes: int = 10000):
        self.storage_path = storage_path
        self.max_episodes = max_episodes
        self._episodes: List[Episode] = []
        self._summaries: List[ExperimentSummary] = []
        if storage_path and os.path.exists(storage_path):
            self._load()

    def record(self, episode: Episode) -> None:
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

    def record_summary(self, summary: ExperimentSummary) -> None:
        self._summaries.append(summary)

    def get_recent(self, n: int = 10) -> List[Episode]:
        return self._episodes[-n:]

    def get_by_action(self, action_name: str, n: int = 10) -> List[Episode]:
        matches = [e for e in self._episodes if e.action == action_name]
        return matches[-n:]

    def get_by_tags(self, tags: List[str], n: int = 10) -> List[Episode]:
        tag_set = set(tags)
        matches = [e for e in self._episodes if tag_set & set(e.tags)]
        return matches[-n:]

    def get_successful_episodes(self, n: int = 10) -> List[Episode]:
        matches = [e for e in self._episodes if e.outcome_quality in ("good", "excellent")]
        return matches[-n:]

    def get_summaries(self, n: int = 5) -> List[ExperimentSummary]:
        return self._summaries[-n:]

    def save(self) -> None:
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "episodes": [e.to_dict() for e in self._episodes],
            "summaries": [s.to_dict() for s in self._summaries],
        }
        with open(self.storage_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            self._episodes = [Episode.from_dict(e) for e in data.get("episodes", [])]
            self._summaries = [ExperimentSummary.from_dict(s) for s in data.get("summaries", [])]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_summaries(self) -> int:
        return len(self._summaries)
