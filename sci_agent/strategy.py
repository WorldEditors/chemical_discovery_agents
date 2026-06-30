from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompoundInfo:
    name: str
    purchasable: bool = False
    price_per_gram: Optional[float] = None
    state_at_room_temp: Optional[str] = None
    melting_point_C: Optional[float] = None
    boiling_point_C: Optional[float] = None
    molecular_weight_approx: Optional[float] = None
    biological_activity: Optional[str] = None
    biological_activity_note: Optional[str] = None
    toxicity_level: Optional[str] = None
    toxicity_note: Optional[str] = None
    is_solvent: bool = False
    solubility_g_per_100mL: Dict[str, float] = field(default_factory=dict)
    produced_by: List[str] = field(default_factory=list)
    used_in: List[str] = field(default_factory=list)

    @property
    def is_medicinal_candidate(self) -> bool:
        if self.biological_activity and self.biological_activity.lower() not in (
            "low", "none", "minimal", "negligible", "undetectable"
        ):
            return True
        return False

    @property
    def is_low_toxicity(self) -> bool:
        if self.toxicity_level and self.toxicity_level.lower() in (
            "low", "medium", "none", "minimal", "negligible", "safe"
        ):
            return True
        return False


@dataclass
class ReactionInfo:
    reactants: List[str]
    products: Dict[str, float]
    byproducts: Dict[str, float] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    conversion: float = 0.0
    cost: Optional[float] = None
    chain_reaction: bool = False
    reactions_count: int = 1
    final_temp_C: Optional[float] = None
    final_pressure_atm: Optional[float] = None
    equipment_used: Optional[str] = None

    @property
    def key(self) -> str:
        return "|".join(sorted(self.reactants))


@dataclass
class SynthesisRoute:
    target: str
    steps: List[Dict[str, Any]]
    estimated_cost: Optional[float] = None
    actual_cost: Optional[float] = None
    actual_yield_g: Optional[float] = None
    viable: bool = True
    notes: List[str] = field(default_factory=list)


class KnowledgeGraph:
    def __init__(self):
        self.compounds: Dict[str, CompoundInfo] = {}
        self.reactions: List[ReactionInfo] = []
        self.failed_combinations: Set[str] = set()
        self.routes: List[SynthesisRoute] = []
        self.total_cost_spent: float = 0.0
        self.total_yield_produced: Dict[str, float] = {}
        self.inventory: Dict[str, float] = {}

    def add_purchasable(self, name: str, info: Dict[str, Any]) -> None:
        if name not in self.compounds:
            self.compounds[name] = CompoundInfo(name=name, purchasable=True)
        c = self.compounds[name]
        c.purchasable = True
        c.price_per_gram = info.get("price_per_gram")
        c.state_at_room_temp = info.get("state_at_room_temp")
        c.molecular_weight_approx = info.get("molecular_weight_approx")
        if info.get("role") == "solvent":
            c.is_solvent = True

    def add_analysis(self, name: str, info: Dict[str, Any]) -> None:
        if name not in self.compounds:
            self.compounds[name] = CompoundInfo(name=name)
        c = self.compounds[name]
        c.melting_point_C = info.get("melting_point_C")
        c.boiling_point_C = info.get("boiling_point_C")
        c.molecular_weight_approx = info.get("molecular_weight_approx", c.molecular_weight_approx)
        c.state_at_room_temp = info.get("state_at_room_temp", c.state_at_room_temp)
        c.biological_activity = info.get("biological_activity")
        c.biological_activity_note = info.get("biological_activity_note")
        c.toxicity_level = info.get("toxicity_level")
        c.toxicity_note = info.get("toxicity_note")
        if info.get("role") == "solvent":
            c.is_solvent = True
        if info.get("solubility_g_per_100mL"):
            c.solubility_g_per_100mL = info["solubility_g_per_100mL"]

    def add_reaction_result(self, reactants: List[str], result: Dict[str, Any]) -> None:
        products = result.get("products_g", {})
        byproducts = result.get("byproducts_g", {})
        rxn = ReactionInfo(
            reactants=sorted(reactants),
            products=products,
            byproducts=byproducts,
            conditions={
                "temperature_C": result.get("_temperature_C"),
                "pressure_atm": result.get("_pressure_atm"),
                "duration_seconds": result.get("_duration_seconds"),
                "equipment": result.get("equipment_used"),
            },
            conversion=result.get("conversion", 0.0),
            cost=result.get("cost", {}).get("total_cost") if isinstance(result.get("cost"), dict) else None,
            chain_reaction=result.get("chain_reaction", False),
            reactions_count=result.get("reactions_count", 1),
            final_temp_C=result.get("final_temperature_C"),
            final_pressure_atm=result.get("final_pressure_atm"),
            equipment_used=result.get("equipment_used"),
        )
        self.reactions.append(rxn)

        for prod_name in products:
            if prod_name not in self.compounds:
                self.compounds[prod_name] = CompoundInfo(name=prod_name)
            self.compounds[prod_name].produced_by.append(rxn.key)
        for prod_name in byproducts:
            if prod_name not in self.compounds:
                self.compounds[prod_name] = CompoundInfo(name=prod_name)

        for reactant in reactants:
            if reactant in self.compounds:
                self.compounds[reactant].used_in.append(rxn.key)

    def add_failed_combination(self, reactants: List[str]) -> None:
        key = "|".join(sorted(reactants))
        self.failed_combinations.add(key)

    def is_tried(self, reactants: List[str]) -> bool:
        key = "|".join(sorted(reactants))
        if key in self.failed_combinations:
            return True
        for rxn in self.reactions:
            if rxn.key == key:
                return True
        return False

    def get_purchasable(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if c.purchasable]

    def get_solvents(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if c.is_solvent]

    def get_non_solvent_purchasable(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if c.purchasable and not c.is_solvent]

    def get_best_solvent_for(self, compound_names: List[str], max_temp_C: float = 200.0) -> Optional[CompoundInfo]:
        solvents = self.get_solvents()
        best = None
        best_bp = -273.0
        for s in solvents:
            if s.boiling_point_C is not None and s.boiling_point_C <= max_temp_C:
                continue
            all_dissolve = all(
                s.name in self.compounds.get(n, CompoundInfo(name=n)).solubility_g_per_100mL
                or self.compounds.get(n, CompoundInfo(name=n)).is_solvent
                for n in compound_names
            )
            if not all_dissolve:
                all_dissolve = True
                for n in compound_names:
                    c = self.compounds.get(n)
                    if c and not c.is_solvent and s.name not in c.solubility_g_per_100mL:
                        all_dissolve = False
                        break
            if all_dissolve:
                bp = s.boiling_point_C or 100.0
                if bp > best_bp:
                    best = s
                    best_bp = bp
        return best

    def get_max_reaction_temp(self, reactant_names: List[str]) -> float:
        solvents = self.get_solvents()
        if not solvents:
            return 200.0
        non_solvent_reactants = [
            n for n in reactant_names
            if n in self.compounds and not self.compounds[n].is_solvent
        ]
        if not non_solvent_reactants:
            return 600.0

        solvent_in_reactants = [
            n for n in reactant_names
            if n in self.compounds and self.compounds[n].is_solvent
        ]
        if solvent_in_reactants:
            bps = [self.compounds[n].boiling_point_C for n in solvent_in_reactants if self.compounds[n].boiling_point_C is not None]
            if bps:
                return max(bps) - 5.0
            return 150.0

        best_bp = None
        for s in solvents:
            bp = s.boiling_point_C
            if bp is None:
                bp = 150.0
            if s.name in reactant_names:
                if best_bp is None or bp > best_bp:
                    best_bp = bp
            else:
                if best_bp is None or bp > best_bp:
                    best_bp = bp
        return (best_bp or 150.0) - 5.0

    def get_synthesized(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if not c.purchasable and c.produced_by]

    def get_unanalyzed(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if c.biological_activity is None and c.produced_by]

    def get_medicinal_candidates(self) -> List[CompoundInfo]:
        return [c for c in self.compounds.values() if c.is_medicinal_candidate]

    def get_qualifying_candidates(self) -> List[CompoundInfo]:
        candidates = [c for c in self.compounds.values() if c.is_medicinal_candidate and c.is_low_toxicity]
        tox_order = {"low": 0, "none": 0, "minimal": 0, "negligible": 0, "safe": 0, "medium": 1}
        candidates.sort(key=lambda c: tox_order.get((c.toxicity_level or "").lower(), 2))
        return candidates

    def get_reactions_producing(self, compound_name: str) -> List[ReactionInfo]:
        return [r for r in self.reactions if compound_name in r.products]

    def get_cheapest_route_to(self, compound_name: str) -> Optional[ReactionInfo]:
        producing = self.get_reactions_producing(compound_name)
        if not producing:
            return None
        costed = [r for r in producing if r.cost is not None]
        if costed:
            return min(costed, key=lambda r: r.cost / max(r.products.get(compound_name, 0.001), 0.001))
        return producing[0]


class ExplorationPhase:
    SURVEY = "survey"
    COMBINATORIAL = "combinatorial"
    ANALYSIS = "analysis"
    ROUTE_DISCOVERY = "route_discovery"
    OPTIMIZATION = "optimization"
    PRODUCTION = "production"
    DONE = "done"


@dataclass
class StrategyState:
    phase: str = ExplorationPhase.SURVEY
    kg: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    combo_queue: List[List[str]] = field(default_factory=list)
    analysis_queue: List[str] = field(default_factory=list)
    route_candidates: List[str] = field(default_factory=list)
    best_target: Optional[str] = None
    best_route: Optional[SynthesisRoute] = None
    submitted: bool = False
    step_count: int = 0
    budget_used: float = 0.0
    time_elapsed: float = 0.0
    time_budget: float = 14400.0


class StrategyEngine:
    CONDITION_PROFILES = [
        {"temperature_C": 80.0, "pressure_atm": 1.0, "duration_seconds": 300.0},
        {"temperature_C": 25.0, "pressure_atm": 1.0, "duration_seconds": 300.0},
        {"temperature_C": 120.0, "pressure_atm": 1.0, "duration_seconds": 600.0},
    ]

    DIAGNOSTIC_DURATION = 300.0
    EXPLORE_AMOUNT = 2.0
    EXPLORE_SOLVENT_MULT = 5.0

    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.state = StrategyState()
        self._combo_size = 2
        self._max_combo_size = 4
        self._condition_index = 0
        self._current_condition_combos: List[List[str]] = []
        self._tried_with_conditions: Dict[str, Set[int]] = {}
        self._diagnostic_done = False
        self._diagnostic_combo: Optional[List[str]] = None
        self._diagnostic_temp_idx = 0
        self._working_conditions: List[int] = []
        self._needs_inventory_check = False
        self._pending_reaction_result: Optional[Dict] = None
        self._pending_reaction_reactants: Optional[List[str]] = None

    def get_next_actions(self) -> List[Dict[str, Any]]:
        if self._needs_inventory_check:
            return [{"action": "get_inventory", "arguments": {}}]

        remaining = self.time_remaining
        kg = self.state.kg

        if remaining < 400 and not self.state.submitted:
            if kg.get_medicinal_candidates() or kg.get_synthesized():
                return self._production_actions()
            return [{"action": "finish_experiment", "arguments": {}}]

        if (remaining < 1500
            and self.state.phase == ExplorationPhase.COMBINATORIAL
            and (kg.get_unanalyzed() or self.state.analysis_queue)):
            self.state.phase = ExplorationPhase.ANALYSIS

        phase = self.state.phase
        if phase == ExplorationPhase.SURVEY:
            return self._survey_actions()
        elif phase == ExplorationPhase.COMBINATORIAL:
            return self._combinatorial_actions()
        elif phase == ExplorationPhase.ANALYSIS:
            return self._analysis_actions()
        elif phase == ExplorationPhase.ROUTE_DISCOVERY:
            return self._route_discovery_actions()
        elif phase == ExplorationPhase.OPTIMIZATION:
            return self._optimization_actions()
        elif phase == ExplorationPhase.PRODUCTION:
            return self._production_actions()
        return [{"action": "finish_experiment", "arguments": {}}]

    def set_time_info(self, elapsed: float, budget: float) -> None:
        self.state.time_elapsed = elapsed
        self.state.time_budget = budget

    @property
    def time_remaining(self) -> float:
        return max(0, self.state.time_budget - self.state.time_elapsed)

    def record_result(self, action: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.state.step_count += 1

        duration = arguments.get("duration_seconds", 0)
        if action == "perform_reaction":
            self.state.time_elapsed += duration
        elif action == "analyze_compound":
            self.state.time_elapsed += 300.0

        if action == "list_purchasable":
            if isinstance(result, dict):
                for name, info in result.items():
                    if isinstance(info, dict):
                        self.state.kg.add_purchasable(name, info)

        elif action == "get_inventory":
            if isinstance(result, dict) and self._needs_inventory_check:
                self._needs_inventory_check = False
                old_names = set(self.state.kg.inventory.keys())
                new_products = {}
                for name, entry in result.items():
                    if name == "success" or not isinstance(entry, dict):
                        continue
                    g = entry.get("amount_g", 0)
                    if isinstance(g, (int, float)) and g > 0:
                        old_g = self.state.kg.inventory.get(name, 0)
                        if name not in old_names or g > old_g + 0.001:
                            diff = g - old_g
                            if diff > 0.001:
                                new_products[name] = round(diff, 4)
                        self.state.kg.inventory[name] = g

                if new_products and self._pending_reaction_result is not None:
                    pending = self._pending_reaction_result
                    pending["products_g"] = new_products
                    pending["byproducts_g"] = {}
                    reactants = self._pending_reaction_reactants
                    self.state.kg.add_reaction_result(reactants, pending)
                    for pname, pg in new_products.items():
                        self.state.kg.total_yield_produced[pname] = self.state.kg.total_yield_produced.get(pname, 0) + pg
                        if pname not in self.state.analysis_queue:
                            if pname not in self.state.kg.compounds or self.state.kg.compounds[pname].biological_activity is None:
                                self.state.analysis_queue.append(pname)
                    self.state.phase = ExplorationPhase.ANALYSIS
                    self._pending_reaction_result = None
                    self._pending_reaction_reactants = None
                elif self._pending_reaction_reactants:
                    self.state.kg.add_failed_combination(self._pending_reaction_reactants)
                    self._pending_reaction_result = None
                    self._pending_reaction_reactants = None

        elif action == "purchase":
            cost = result.get("cost", 0)
            if isinstance(cost, (int, float)):
                self.state.kg.total_cost_spent += cost
            name = arguments.get("chemical_name", "")
            amount = arguments.get("amount_grams", 0)
            if isinstance(result, dict) and result.get("success", True) and "error" not in result:
                self.state.kg.inventory[name] = self.state.kg.inventory.get(name, 0) + amount

        elif action == "analyze_compound":
            name = arguments.get("chemical_name", "")
            if isinstance(result, dict) and result.get("success"):
                name = result.get("name", name)
                self.state.kg.add_analysis(name, result)
            else:
                if name in self.state.analysis_queue:
                    self.state.analysis_queue.remove(name)
                if name in self.state.kg.compounds:
                    self.state.kg.compounds[name].biological_activity = "unknown"

        elif action == "perform_reaction":
            reactants = list(arguments.get("reactant_amounts", {}).keys())
            reactant_amounts = arguments.get("reactant_amounts", {})

            dissolution = result.get("dissolution") if isinstance(result, dict) else None
            if dissolution and isinstance(dissolution, dict):
                for solvent_name, info in dissolution.items():
                    undissolved = info.get("undissolved_g", {})
                    if undissolved:
                        logger.debug(f"Undissolved in {solvent_name}: {undissolved}")

            if isinstance(result, dict) and result.get("success"):
                for rname, ramount in reactant_amounts.items():
                    self.state.kg.inventory[rname] = max(0, self.state.kg.inventory.get(rname, 0) - ramount)

                num_products = result.get("num_products_formed", 0)
                cost = result.get("cost", {})
                if isinstance(cost, dict):
                    self.state.kg.total_cost_spent += cost.get("total_cost", 0)

                if num_products > 0:
                    self._pending_reaction_result = result
                    self._pending_reaction_result["_temperature_C"] = arguments.get("temperature_C")
                    self._pending_reaction_result["_pressure_atm"] = arguments.get("pressure_atm")
                    self._pending_reaction_result["_duration_seconds"] = arguments.get("duration_seconds")
                    self._pending_reaction_reactants = reactants
                    self._needs_inventory_check = True

                    if not self._diagnostic_done:
                        cond_idx = self._diagnostic_temp_idx - 1
                        if cond_idx >= 0 and cond_idx not in self._working_conditions:
                            self._working_conditions.append(cond_idx)
                        self._diagnostic_done = True
                        if self._working_conditions:
                            self._condition_index = self._working_conditions[0]
                        self.state.combo_queue = []
                else:
                    self.state.kg.add_failed_combination(reactants)
            elif isinstance(result, dict) and "Insufficient" in result.get("message", ""):
                pass
            elif isinstance(result, dict) and "time" in result.get("message", "").lower():
                pass
            else:
                for rname, ramount in reactant_amounts.items():
                    if not arguments.get("recover_on_failure"):
                        self.state.kg.inventory[rname] = max(0, self.state.kg.inventory.get(rname, 0) - ramount)
                self.state.kg.add_failed_combination(reactants)

        elif action == "submit_solution":
            if isinstance(result, dict) and result.get("passed"):
                self.state.submitted = True
            elif isinstance(result, dict) and not result.get("passed"):
                target = arguments.get("target_compound", "")
                violations = result.get("violations", [])
                tox_viol = [v for v in violations if "toxicity" in v.lower() or "tox" in v.lower()]
                med_viol = [v for v in violations if "medicin" in v.lower()]
                yield_viol = [v for v in violations if "yield" in v.lower()]
                if tox_viol or med_viol:
                    if target in self.state.kg.compounds:
                        c = self.state.kg.compounds[target]
                        if tox_viol:
                            c.toxicity_level = "high"
                        if med_viol:
                            c.biological_activity = "low"
                    self.state.phase = ExplorationPhase.COMBINATORIAL
                    purchasable = sorted([c.name for c in self.state.kg.get_purchasable()])
                    if self._condition_index < len(self.CONDITION_PROFILES) - 1:
                        self._condition_index += 1
                        self.state.combo_queue = [purchasable]
                    else:
                        self.state.phase = ExplorationPhase.DONE
                elif yield_viol:
                    time_remaining = self.state.time_budget - self.state.time_elapsed
                    if time_remaining < 120:
                        self.state.phase = ExplorationPhase.DONE
                elif not yield_viol:
                    self.state.submitted = True

        self._update_phase()

    def _update_phase(self) -> None:
        kg = self.state.kg
        phase = self.state.phase

        if phase == ExplorationPhase.SURVEY:
            if kg.get_purchasable():
                self._build_combo_queue()
                self.state.phase = ExplorationPhase.COMBINATORIAL

        elif phase == ExplorationPhase.COMBINATORIAL:
            unanalyzed = kg.get_unanalyzed()
            if unanalyzed:
                self.state.phase = ExplorationPhase.ANALYSIS
                return
            if not self.state.combo_queue:
                if self.state.analysis_queue:
                    self.state.phase = ExplorationPhase.ANALYSIS
                elif kg.get_synthesized() and not kg.get_qualifying_candidates():
                    self.state.phase = ExplorationPhase.ANALYSIS
                elif self._combo_size < self._max_combo_size:
                    self._combo_size += 1
                    self._build_combo_queue()
                    if not self.state.combo_queue:
                        if self._try_next_condition_sweep():
                            pass
                        elif kg.get_qualifying_candidates():
                            self.state.phase = ExplorationPhase.PRODUCTION
                        else:
                            self._expand_with_products()
                            if not self.state.combo_queue:
                                if kg.get_medicinal_candidates():
                                    self.state.phase = ExplorationPhase.PRODUCTION
                                else:
                                    self.state.phase = ExplorationPhase.DONE
                elif self._try_next_condition_sweep():
                    pass
                elif kg.get_qualifying_candidates():
                    self.state.phase = ExplorationPhase.PRODUCTION
                else:
                    self._expand_with_products()
                    if not self.state.combo_queue:
                        if kg.get_medicinal_candidates():
                            self.state.phase = ExplorationPhase.PRODUCTION
                        else:
                            self.state.phase = ExplorationPhase.DONE

        elif phase == ExplorationPhase.ANALYSIS:
            if not self.state.analysis_queue and not kg.get_unanalyzed():
                if kg.get_qualifying_candidates():
                    self.state.phase = ExplorationPhase.PRODUCTION
                elif kg.get_medicinal_candidates():
                    self.state.phase = ExplorationPhase.PRODUCTION
                else:
                    purchasable = sorted([c.name for c in kg.get_purchasable()])
                    if self._condition_index < len(self.CONDITION_PROFILES) - 1:
                        self._condition_index += 1
                        self.state.combo_queue = [purchasable]
                        self.state.phase = ExplorationPhase.COMBINATORIAL
                    else:
                        self._expand_with_products()
                        if self.state.combo_queue:
                            self.state.phase = ExplorationPhase.COMBINATORIAL
                        else:
                            self.state.phase = ExplorationPhase.DONE

        elif phase == ExplorationPhase.ROUTE_DISCOVERY:
            if kg.get_qualifying_candidates():
                self.state.phase = ExplorationPhase.PRODUCTION
            elif not self.state.route_candidates:
                self._expand_with_products()
                if self.state.combo_queue:
                    self.state.phase = ExplorationPhase.COMBINATORIAL
                else:
                    self.state.phase = ExplorationPhase.DONE

        elif phase == ExplorationPhase.OPTIMIZATION:
            if self.state.best_target:
                self.state.phase = ExplorationPhase.PRODUCTION

        elif phase == ExplorationPhase.PRODUCTION:
            if self.state.submitted:
                self.state.phase = ExplorationPhase.DONE

    def _build_combo_queue(self) -> None:
        purchasable = [c.name for c in self.state.kg.get_purchasable()]
        solvents = [c.name for c in self.state.kg.get_solvents()]
        non_solvents = [c.name for c in self.state.kg.get_non_solvent_purchasable()]
        self._max_combo_size = min(5, len(purchasable))

        all_combos = []
        if solvents and non_solvents:
            for size in range(2, min(5, len(non_solvents) + 1)):
                for ns_combo in itertools.combinations(non_solvents, size):
                    for solvent in solvents:
                        combo = sorted(list(ns_combo) + [solvent])
                        if not self.state.kg.is_tried(combo):
                            all_combos.append(combo)
            for size in range(2, min(4, len(non_solvents) + 1)):
                for ns_combo in itertools.combinations(non_solvents, size):
                    combo = sorted(list(ns_combo))
                    if not self.state.kg.is_tried(combo):
                        all_combos.append(combo)
        else:
            for size in range(2, self._max_combo_size + 1):
                combos = list(itertools.combinations(purchasable, size))
                untried = [list(c) for c in combos if not self.state.kg.is_tried(list(c))]
                all_combos.extend(untried)

        self.state.combo_queue = all_combos[:40]
        self._combo_size = self._max_combo_size

    def _expand_with_products(self) -> None:
        purchasable = [c.name for c in self.state.kg.get_purchasable()]
        synthesized = [c.name for c in self.state.kg.get_synthesized()]

        self._replenish_queue = []
        for sname in synthesized:
            inv = self.state.kg.inventory.get(sname, 0)
            if inv < 1.0:
                rxns = self.state.kg.get_reactions_producing(sname)
                if rxns:
                    best = rxns[0]
                    self._replenish_queue.append(best.reactants)

        new_combos = []

        all_synth_combo = sorted(synthesized + purchasable)
        if len(all_synth_combo) <= 8:
            new_combos.append(all_synth_combo)

        for n_synth in range(len(synthesized), 0, -1):
            for synth_combo in itertools.combinations(synthesized, n_synth):
                for n_purch in range(len(purchasable), 0, -1):
                    for purch_combo in itertools.combinations(purchasable, n_purch):
                        combo = sorted(list(synth_combo) + list(purch_combo))
                        if 2 <= len(combo) <= 8 and not self.state.kg.is_tried(combo):
                            new_combos.append(combo)

        for s1, s2 in itertools.combinations(synthesized, 2):
            combo = sorted([s1, s2])
            if not self.state.kg.is_tried(combo):
                new_combos.append(combo)

        self.state.combo_queue = new_combos[:80]
        self._condition_index = 0
        self._tried_with_conditions = {}

    def _survey_actions(self) -> List[Dict[str, Any]]:
        return [
            {"action": "list_purchasable", "arguments": {}},
            {"action": "list_equipment", "arguments": {}},
        ]

    def _combinatorial_actions(self) -> List[Dict[str, Any]]:
        if hasattr(self, '_replenish_queue') and self._replenish_queue:
            reactants = self._replenish_queue.pop(0)
            actions = []
            amounts = {}
            for name in reactants:
                c = self.state.kg.compounds.get(name)
                if c and c.purchasable:
                    actions.append({
                        "action": "purchase",
                        "arguments": {"chemical_name": name, "amount_grams": 3.0},
                    })
                    amounts[name] = 3.0
                else:
                    available = self.state.kg.inventory.get(name, 0)
                    if available < 0.1:
                        amounts[name] = 0.1
                    else:
                        amounts[name] = min(2.0, available * 0.5)
            conditions = self.CONDITION_PROFILES[self._condition_index if self._working_conditions else 0]
            rxn_args = {
                "reactant_amounts": amounts,
                "temperature_C": conditions["temperature_C"],
                "pressure_atm": conditions["pressure_atm"],
                "duration_seconds": conditions["duration_seconds"],
                "recover_on_failure": False,
            }
            actions.append({"action": "perform_reaction", "arguments": rxn_args})
            return actions

        if not self._diagnostic_done:
            return self._diagnostic_actions()

        if not self.state.combo_queue:
            if self._try_next_condition_sweep():
                return self._combinatorial_actions()
            self._update_phase()
            return self.get_next_actions()

        combo = self.state.combo_queue.pop(0)
        combo_key = "|".join(sorted(combo))

        if combo_key not in self._tried_with_conditions:
            self._tried_with_conditions[combo_key] = set()
        self._tried_with_conditions[combo_key].add(self._condition_index)

        conditions = self.CONDITION_PROFILES[self._condition_index]

        purchasable_names = set(c.name for c in self.state.kg.get_purchasable())
        is_full_diagnostic = all(n in purchasable_names for n in combo) and len(combo) >= len(purchasable_names) - 1
        if is_full_diagnostic:
            explore_amount = self.EXPLORE_AMOUNT
        else:
            explore_amount = self.EXPLORE_AMOUNT

        actions = []
        amounts = {}
        skip = False
        needs_replenish = None
        for name in combo:
            c = self.state.kg.compounds.get(name)
            if c and c.purchasable:
                actions.append({
                    "action": "purchase",
                    "arguments": {"chemical_name": name, "amount_grams": explore_amount},
                })
                amounts[name] = explore_amount
            else:
                available = self.state.kg.inventory.get(name, 0)
                if available < 0.05:
                    needs_replenish = name
                    skip = True
                    break
                amounts[name] = min(0.5, available * 0.8)

        if skip:
            if needs_replenish:
                rxns = self.state.kg.get_reactions_producing(needs_replenish)
                if rxns:
                    best = rxns[0]
                    replenish_actions = []
                    replenish_amounts = {}
                    for rname in best.reactants:
                        rc = self.state.kg.compounds.get(rname)
                        if rc and rc.purchasable:
                            replenish_actions.append({
                                "action": "purchase",
                                "arguments": {"chemical_name": rname, "amount_grams": 3.0},
                            })
                            replenish_amounts[rname] = 3.0
                        else:
                            avail = self.state.kg.inventory.get(rname, 0)
                            if avail < 0.1:
                                return self._combinatorial_actions()
                            replenish_amounts[rname] = min(2.0, avail * 0.8)
                    conds = best.conditions or {}
                    temp = conds.get("temperature_C") or 80.0
                    pressure = conds.get("pressure_atm") or 1.0
                    duration = max(conds.get("duration_seconds") or 300.0, 300.0)
                    replenish_actions.append({
                        "action": "perform_reaction",
                        "arguments": {
                            "reactant_amounts": replenish_amounts,
                            "temperature_C": temp,
                            "pressure_atm": pressure,
                            "duration_seconds": duration,
                            "recover_on_failure": False,
                        },
                    })
                    self.state.combo_queue.insert(0, combo)
                    return replenish_actions
            return self._combinatorial_actions()

        temp_C = conditions["temperature_C"]
        max_temp = self.state.kg.get_max_reaction_temp(combo)
        if temp_C > max_temp:
            temp_C = max(25.0, max_temp)

        for name in combo:
            c = self.state.kg.compounds.get(name)
            if c and c.is_solvent:
                amounts[name] = max(amounts.get(name, 0), explore_amount * self.EXPLORE_SOLVENT_MULT)

        equipment = None
        if conditions["pressure_atm"] > 1.0:
            equipment = "sealed_flask"

        duration = self.DIAGNOSTIC_DURATION if is_full_diagnostic else conditions["duration_seconds"]
        rxn_args = {
            "reactant_amounts": amounts,
            "temperature_C": temp_C,
            "pressure_atm": conditions["pressure_atm"],
            "duration_seconds": duration,
            "recover_on_failure": False,
        }
        if equipment:
            rxn_args["equipment"] = equipment

        actions.append({"action": "perform_reaction", "arguments": rxn_args})
        return actions

    def _diagnostic_actions(self) -> List[Dict[str, Any]]:
        purchasable = [c.name for c in self.state.kg.get_purchasable()]
        if not purchasable or len(purchasable) < 2:
            self._diagnostic_done = True
            return self._combinatorial_actions()

        if self._diagnostic_combo is None:
            self._diagnostic_combo = sorted(purchasable)

        if self._diagnostic_temp_idx >= len(self.CONDITION_PROFILES):
            self._diagnostic_done = True
            if self._working_conditions:
                self._condition_index = self._working_conditions[0]
            else:
                self._condition_index = 0
            self._build_combo_queue()
            return self._combinatorial_actions()

        conditions = self.CONDITION_PROFILES[self._diagnostic_temp_idx]
        self._diagnostic_temp_idx += 1

        diag_amount = self.EXPLORE_AMOUNT

        actions = []
        amounts = {}
        for name in self._diagnostic_combo:
            c = self.state.kg.compounds.get(name)
            amt = diag_amount * self.EXPLORE_SOLVENT_MULT if (c and c.is_solvent) else diag_amount
            actions.append({
                "action": "purchase",
                "arguments": {"chemical_name": name, "amount_grams": amt},
            })
            amounts[name] = amt

        temp_C = conditions["temperature_C"]
        max_temp = self.state.kg.get_max_reaction_temp(self._diagnostic_combo)
        if temp_C > max_temp:
            temp_C = max(25.0, max_temp)

        equipment = None
        if conditions["pressure_atm"] > 1.0:
            equipment = "sealed_flask"

        rxn_args = {
            "reactant_amounts": amounts,
            "temperature_C": temp_C,
            "pressure_atm": conditions["pressure_atm"],
            "duration_seconds": self.DIAGNOSTIC_DURATION,
            "recover_on_failure": False,
        }
        if equipment:
            rxn_args["equipment"] = equipment

        actions.append({"action": "perform_reaction", "arguments": rxn_args})
        return actions

    def _try_next_condition_sweep(self) -> bool:
        if self._condition_index >= len(self.CONDITION_PROFILES) - 1:
            return False

        self._condition_index += 1

        purchasable = [c.name for c in self.state.kg.get_purchasable()]
        synthesized = [c.name for c in self.state.kg.get_synthesized()]
        all_available = purchasable + synthesized

        new_queue = []

        if synthesized:
            for s in synthesized:
                for n in range(1, min(4, len(purchasable)) + 1):
                    for pcombo in itertools.combinations(purchasable, n):
                        combo = sorted([s] + list(pcombo))
                        combo_key = "|".join(combo)
                        tried = self._tried_with_conditions.get(combo_key, set())
                        if self._condition_index not in tried:
                            already_reacted = any(
                                r for r in self.state.kg.reactions
                                if r.key == combo_key and r.products
                            )
                            if not already_reacted:
                                new_queue.append(combo)

        for size in range(2, self._max_combo_size + 1):
            combos = list(itertools.combinations(purchasable, size))
            for c in combos:
                combo_key = "|".join(sorted(c))
                tried = self._tried_with_conditions.get(combo_key, set())
                if self._condition_index not in tried:
                    already_reacted = any(
                        r for r in self.state.kg.reactions
                        if r.key == combo_key and r.products
                    )
                    if not already_reacted:
                        new_queue.append(list(c))

        if new_queue:
            self.state.combo_queue = new_queue[:30]
            return True
        return False

    def _analysis_actions(self) -> List[Dict[str, Any]]:
        if not self.state.analysis_queue:
            unanalyzed = self.state.kg.get_unanalyzed()
            if unanalyzed:
                self.state.analysis_queue = [c.name for c in unanalyzed]
            else:
                self._update_phase()
                return self.get_next_actions()

        name = self.state.analysis_queue.pop(0)
        return [{"action": "analyze_compound", "arguments": {"chemical_name": name}}]

    def _route_discovery_actions(self) -> List[Dict[str, Any]]:
        candidates = self.state.kg.get_medicinal_candidates()
        if not candidates:
            self.state.phase = ExplorationPhase.DONE
            return self.get_next_actions()

        self.state.route_candidates = [c.name for c in candidates]
        self.state.phase = ExplorationPhase.PRODUCTION
        self.state.best_target = candidates[0].name
        return self.get_next_actions()

    def _optimization_actions(self) -> List[Dict[str, Any]]:
        return self._production_actions()

    def _production_actions(self) -> List[Dict[str, Any]]:
        qualifying = self.state.kg.get_qualifying_candidates()
        if qualifying:
            target = qualifying[0]
        else:
            medicinal = self.state.kg.get_medicinal_candidates()
            if medicinal:
                target = medicinal[0]
            else:
                self.state.phase = ExplorationPhase.DONE
                return [{"action": "finish_experiment", "arguments": {}}]

        self.state.best_target = target.name

        min_yield = self.constraints.get("min_yield_g", 1.0)
        current_yield = self.state.kg.total_yield_produced.get(target.name, 0)

        if current_yield < min_yield:
            time_remaining = self.state.time_budget - self.state.time_elapsed
            if time_remaining < 60 and current_yield > 0:
                return [{"action": "submit_solution", "arguments": {"target_compound": target.name}}]

            rxns = self.state.kg.get_reactions_producing(target.name)
            if not rxns:
                self.state.phase = ExplorationPhase.DONE
                return [{"action": "finish_experiment", "arguments": {}}]

            best_rxn = max(rxns, key=lambda r: r.products.get(target.name, 0))
            deficit = min_yield - current_yield
            scale = max(1.0, deficit / max(best_rxn.products.get(target.name, 0.1), 0.01))
            scale = min(scale, 10.0)

            actions = []
            amounts = {}
            for rname in best_rxn.reactants:
                c = self.state.kg.compounds.get(rname)
                if c and c.purchasable:
                    buy_amount = round(3.0 * scale, 1)
                    if c.is_solvent:
                        buy_amount = round(15.0 * scale, 1)
                    actions.append({
                        "action": "purchase",
                        "arguments": {"chemical_name": rname, "amount_grams": buy_amount},
                    })
                    amounts[rname] = buy_amount
                else:
                    available = self.state.kg.inventory.get(rname, 0)
                    needed = round(2.0 * scale, 1)
                    if available < needed:
                        sub_rxns = self.state.kg.get_reactions_producing(rname)
                        if sub_rxns:
                            pass
                    amounts[rname] = min(needed, max(0.1, available * 0.9))

            has_solvent_in_rxn = any(
                self.state.kg.compounds.get(n, CompoundInfo(name=n)).is_solvent
                for n in best_rxn.reactants
            )
            if not has_solvent_in_rxn:
                best_solvent = self.state.kg.get_best_solvent_for(
                    best_rxn.reactants, max_temp_C=80.0
                )
                if best_solvent:
                    sol_amount = round(15.0 * scale, 1)
                    actions.append({
                        "action": "purchase",
                        "arguments": {"chemical_name": best_solvent.name, "amount_grams": sol_amount},
                    })
                    amounts[best_solvent.name] = sol_amount

            conditions = best_rxn.conditions or {}
            temp = conditions.get("temperature_C") or 80.0
            pressure = conditions.get("pressure_atm") or 1.0
            duration = conditions.get("duration_seconds") or 300.0
            max_temp = self.state.kg.get_max_reaction_temp(list(amounts.keys()))
            if temp > max_temp:
                temp = max(25.0, max_temp)
            time_remaining = self.state.time_budget - self.state.time_elapsed
            if time_remaining < duration + 100:
                duration = max(60.0, time_remaining - 50)

            actions.append({
                "action": "perform_reaction",
                "arguments": {
                    "reactant_amounts": amounts,
                    "temperature_C": temp,
                    "pressure_atm": pressure,
                    "duration_seconds": duration,
                    "recover_on_failure": False,
                },
            })
            return actions

        return [{"action": "submit_solution", "arguments": {"target_compound": target.name}}]

    def get_knowledge_summary(self) -> str:
        kg = self.state.kg
        lines = [
            f"Phase: {self.state.phase}",
            f"Compounds known: {len(kg.compounds)} ({len(kg.get_purchasable())} purchasable, {len(kg.get_synthesized())} synthesized)",
            f"Reactions discovered: {len(kg.reactions)}",
            f"Failed combos: {len(kg.failed_combinations)}",
            f"Analysis queue: {len(self.state.analysis_queue)}",
            f"Combo queue: {len(self.state.combo_queue)}",
            f"Total cost spent: {kg.total_cost_spent:.1f}",
        ]
        med = kg.get_medicinal_candidates()
        if med:
            lines.append(f"Medicinal candidates: {[c.name for c in med[:5]]}")
        qual = kg.get_qualifying_candidates()
        if qual:
            lines.append(f"Qualifying (med+low_tox): {[c.name for c in qual[:5]]}")
        return "\n".join(lines)
