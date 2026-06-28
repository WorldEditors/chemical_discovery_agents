from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set


@dataclass
class CompoundKnowledge:
    """Accumulated knowledge about a compound discovered through experiments."""

    compound_id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    known_as_product_of: List[str] = field(default_factory=list)
    known_as_reactant_in: List[str] = field(default_factory=list)
    medicinal_hints: List[str] = field(default_factory=list)
    toxicity_hints: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompoundKnowledge:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ReactionKnowledge:
    """Accumulated knowledge about a reaction."""

    reaction_id: str
    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    observed_conditions: List[Dict[str, Any]] = field(default_factory=list)
    yield_observations: List[Dict[str, Any]] = field(default_factory=list)
    cost_observations: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReactionKnowledge:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StrategyKnowledge:
    """High-level strategic insights accumulated across sessions."""

    insight: str
    confidence: float = 0.5
    source_sessions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyKnowledge:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SemanticMemory:
    """Long-term structured knowledge base about the chemistry world.

    Stores discovered facts about compounds, reactions, and strategic insights.
    Persists across sessions to enable cumulative learning.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._compounds: Dict[str, CompoundKnowledge] = {}
        self._reactions: Dict[str, ReactionKnowledge] = {}
        self._strategies: List[StrategyKnowledge] = []
        self._world_id: Optional[str] = None
        if storage_path and os.path.exists(storage_path):
            self._load()

    def set_world(self, world_id: str) -> None:
        self._world_id = world_id

    def update_compound(self, compound_id: str, **kwargs) -> CompoundKnowledge:
        if compound_id not in self._compounds:
            self._compounds[compound_id] = CompoundKnowledge(
                compound_id=compound_id,
                name=kwargs.pop("name", compound_id),
            )
        ck = self._compounds[compound_id]
        for key, val in kwargs.items():
            if hasattr(ck, key):
                attr = getattr(ck, key)
                if isinstance(attr, list) and isinstance(val, (str, dict)):
                    if val not in attr:
                        attr.append(val)
                elif isinstance(attr, list) and isinstance(val, list):
                    for item in val:
                        if item not in attr:
                            attr.append(item)
                elif isinstance(attr, dict) and isinstance(val, dict):
                    attr.update(val)
                else:
                    setattr(ck, key, val)
        return ck

    def update_reaction(self, reaction_id: str, **kwargs) -> ReactionKnowledge:
        if reaction_id not in self._reactions:
            self._reactions[reaction_id] = ReactionKnowledge(reaction_id=reaction_id)
        rk = self._reactions[reaction_id]
        for key, val in kwargs.items():
            if hasattr(rk, key):
                attr = getattr(rk, key)
                if isinstance(attr, list) and isinstance(val, (str, dict)):
                    if val not in attr:
                        attr.append(val)
                elif isinstance(attr, list) and isinstance(val, list):
                    for item in val:
                        if item not in attr:
                            attr.append(item)
                else:
                    setattr(rk, key, val)
        return rk

    def add_strategy(self, insight: str, confidence: float = 0.5, **kwargs) -> None:
        self._strategies.append(StrategyKnowledge(
            insight=insight, confidence=confidence, **kwargs
        ))

    def get_compound(self, compound_id: str) -> Optional[CompoundKnowledge]:
        return self._compounds.get(compound_id)

    def get_reaction(self, reaction_id: str) -> Optional[ReactionKnowledge]:
        return self._reactions.get(reaction_id)

    def get_all_compounds(self) -> List[CompoundKnowledge]:
        return list(self._compounds.values())

    def get_all_reactions(self) -> List[ReactionKnowledge]:
        return list(self._reactions.values())

    def get_strategies(self, min_confidence: float = 0.0) -> List[StrategyKnowledge]:
        return [s for s in self._strategies if s.confidence >= min_confidence]

    def get_compounds_with_medicinal_hints(self) -> List[CompoundKnowledge]:
        return [c for c in self._compounds.values() if c.medicinal_hints]

    def summarize(self) -> str:
        lines = [
            f"Known compounds: {len(self._compounds)}",
            f"Known reactions: {len(self._reactions)}",
            f"Strategic insights: {len(self._strategies)}",
        ]
        medicinal = self.get_compounds_with_medicinal_hints()
        if medicinal:
            lines.append(f"Compounds with medicinal potential: {[c.name for c in medicinal[:5]]}")
        high_conf = self.get_strategies(min_confidence=0.7)
        if high_conf:
            lines.append("Key strategies:")
            for s in high_conf[-5:]:
                lines.append(f"  - {s.insight} (conf={s.confidence:.1f})")
        return "\n".join(lines)

    def save(self) -> None:
        if not self.storage_path:
            return
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = {
            "world_id": self._world_id,
            "compounds": {k: v.to_dict() for k, v in self._compounds.items()},
            "reactions": {k: v.to_dict() for k, v in self._reactions.items()},
            "strategies": [s.to_dict() for s in self._strategies],
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                with open(self.storage_path, encoding=encoding) as f:
                    data = json.load(f)
                self._world_id = data.get("world_id")
                self._compounds = {
                    k: CompoundKnowledge.from_dict(v)
                    for k, v in data.get("compounds", {}).items()
                }
                self._reactions = {
                    k: ReactionKnowledge.from_dict(v)
                    for k, v in data.get("reactions", {}).items()
                }
                self._strategies = [
                    StrategyKnowledge.from_dict(s) for s in data.get("strategies", [])
                ]
                return
            except (UnicodeDecodeError, json.JSONDecodeError, OSError, KeyError):
                continue
