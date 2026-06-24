#!/usr/bin/env python3
"""Evaluation script: runs the agent against xenoverse.chemverse and reports results.

40 pre-sampled worlds (20 easy, 20 medium), each run 3 times.
Reports: avg_score, pass@1, pass@3, pass^3 per difficulty and overall.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_xenoverse_root = os.environ.get("XENOVERSE_ROOT", os.path.join(os.path.dirname(__file__), "..", "Xenoverse"))
if os.path.isdir(_xenoverse_root):
    sys.path.insert(0, os.path.abspath(_xenoverse_root))

from xenoverse.chemverse.environment.backend import SciResearchBackend
from xenoverse.chemverse.task_sampler import SciResearchTaskSampler
from sci_agent import SciResearchAgent, AgentConfig
from sci_agent.tools.env_adapter import EnvironmentToolAdapter

logger = logging.getLogger(__name__)

WORLDS_PER_DIFFICULTY = 20
RUNS_PER_WORLD = 3
DIFFICULTIES = ["easy", "medium"]
BASE_SEEDS = {"easy": 1000, "medium": 2000}

UNSOLVABLE_BASELINE_COST = {"easy": 50.0, "medium": 100.0}


def _fmt_g(g: float) -> str:
    if g == 0:
        return "0g"
    if abs(g) >= 1.0:
        return f"{g:.4f}g"
    if abs(g) >= 0.01:
        return f"{g:.5f}g"
    return f"{g:.4e}g"


def _print_optimal_route(eval_result: Dict[str, Any]) -> None:
    pathway = eval_result.get("optimal_pathway")
    if not pathway:
        print("  [No optimal pathway found]")
        return

    print(f"\n  --- Ground Truth Optimal Route ---")
    print(f"  Target: {pathway.get('target')} (med={pathway.get('medicinal_value'):.4f}, tox={pathway.get('base_toxicity')})")
    print(f"  Total cost: {pathway.get('optimal_cost')} | Yield: {pathway.get('yield_g')}g | Time: {pathway.get('total_time_seconds')}s")
    print(f"  Cost breakdown: purchase={pathway.get('purchase_cost')} + process={pathway.get('process_cost')} + purification={pathway.get('purification_cost')}")
    print(f"  M1 scale: {pathway.get('m1_scale_g')}g per L1 reactant")

    steps = pathway.get("steps", [])
    if steps:
        print(f"  Synthesis steps ({len(steps)}):")
        for i, step in enumerate(steps, 1):
            reactants_str = ", ".join(f"{name}: {_fmt_g(g)}" for name, g in step.get("reactants_g", {}).items())
            catalysts = step.get("catalysts_g", {})
            catalysts_str = ", ".join(f"{name}: {_fmt_g(g)}" for name, g in catalysts.items()) if catalysts else ""
            products_str = ", ".join(f"{name}: {_fmt_g(g)}" for name, g in step.get("products_g", {}).items())
            print(f"    Step {i}: {step.get('reaction_id')} @ {step.get('temperature_C')}°C for {step.get('duration_s')}s")
            print(f"           reactants: {reactants_str}")
            if catalysts_str:
                print(f"           catalysts: {catalysts_str}")
            print(f"           conv={step.get('conversion'):.4f} | process_cost={step.get('process_cost')} | purif_cost={step.get('purification_cost')}")
            print(f"           products: {products_str}")
    print(f"  {'—'*40}")


def get_world_list() -> List[Dict[str, Any]]:
    """Return list of 40 world specs: [{difficulty, seed, world_idx}, ...]"""
    worlds = []
    idx = 0
    for diff in DIFFICULTIES:
        base = BASE_SEEDS[diff]
        for i in range(WORLDS_PER_DIFFICULTY):
            worlds.append({
                "world_idx": idx,
                "difficulty": diff,
                "seed": base + i,
            })
            idx += 1
    return worlds


def presample_task(difficulty: str, seed: int) -> Dict[str, Any]:
    """Pre-sample a task dict for the given difficulty and seed."""
    return SciResearchTaskSampler(
        seed=seed,
        complexity_level=difficulty,
        use_backward_design=True,
    )


def run_single_trial(
    config: AgentConfig,
    task: Dict[str, Any],
    trial_idx: int,
    difficulty: str = "easy",
) -> Dict[str, Any]:
    """Run agent once on a pre-sampled task. Returns result dict with score."""
    trial_config = AgentConfig.from_dict(config.to_dict())
    trial_config.memory_dir = None

    is_solvable = task.get("is_solvable", True)

    backend = SciResearchBackend()
    adapter = EnvironmentToolAdapter(backend=backend)
    agent = SciResearchAgent(config=trial_config, env_adapter=adapter)

    start_time = time.time()
    result = agent.run(task=task, close_on_finish=False)
    elapsed = time.time() - start_time

    session_id = adapter.session_id
    agent_best_cost = result["best_score"]

    trial_result: Dict[str, Any] = {
        "trial_idx": trial_idx,
        "agent_best_cost": agent_best_cost,
        "agent_passed": agent_best_cost is not None,
        "steps_taken": result["steps_taken"],
        "elapsed_seconds": round(elapsed, 2),
        "is_solvable": is_solvable,
    }

    declared_no_solution = False
    total_experiment_cost = 0.0
    optimal_cost = None

    if session_id:
        try:
            env = backend.get_session(session_id)
            declared_no_solution = getattr(env, "_declared_no_solution", False)
            total_experiment_cost = env._total_cost
            constraints = env._task.get("constraints", {}) if env._task else {}
            trial_result["total_experiment_cost"] = round(total_experiment_cost, 2)
            trial_result["elapsed_sim_time"] = round(env._elapsed_time, 1)
            trial_result["declared_no_solution"] = declared_no_solution

            if is_solvable:
                max_tox = constraints.get("max_toxicity", 4.0)
                min_med = constraints.get("min_medicinal", 1.0)
                min_yield = constraints.get("min_yield_g", 0.5)

                optimal = backend.eval_compute_optimal_cost(
                    session_id,
                    min_medicinal_value=min_med,
                    max_toxicity=max_tox,
                    min_yield_g=min_yield,
                )
                if optimal and optimal.get("found"):
                    optimal_cost = optimal.get("optimal_cost")
                    trial_result["optimal_cost"] = optimal_cost
        except Exception as e:
            logger.warning(f"  Could not compute optimal: {e}")

        adapter.close()

    if not is_solvable:
        if declared_no_solution:
            baseline = UNSOLVABLE_BASELINE_COST.get(difficulty, 50.0)
            actual_cost = max(total_experiment_cost, 0.01)
            score = min(1.0, baseline / actual_cost)
        else:
            score = 0.0
        trial_result["score"] = round(score, 4)
        trial_result["agent_passed"] = declared_no_solution
    else:
        if declared_no_solution:
            trial_result["score"] = 0.0
            trial_result["agent_passed"] = False
        elif agent_best_cost is not None and optimal_cost is not None and optimal_cost > 0:
            score = min(1.0, optimal_cost / agent_best_cost)
            trial_result["score"] = round(score, 4)
        else:
            trial_result["score"] = 0.0

    return trial_result


def evaluate_world(
    config: AgentConfig,
    world_spec: Dict[str, Any],
    task: Dict[str, Any],
    n_runs: int = RUNS_PER_WORLD,
) -> Dict[str, Any]:
    """Run n_runs trials on a single world and collect results."""
    world_idx = world_spec["world_idx"]
    difficulty = world_spec["difficulty"]
    seed = world_spec["seed"]

    is_solvable = task.get("is_solvable", True)
    solvable_label = "" if is_solvable else " [UNSOLVABLE]"

    print(f"\n{'='*70}")
    print(f"World {world_idx:02d} | {difficulty} | seed={seed}{solvable_label}")
    print(f"{'='*70}")

    trials = []
    for t in range(n_runs):
        print(f"\n  --- Trial {t+1}/{n_runs} ---")
        trial_result = run_single_trial(config, task, trial_idx=t, difficulty=difficulty)
        trials.append(trial_result)
        if not is_solvable:
            status = "CORRECT_NO_SOL" if trial_result.get("declared_no_solution") else "MISSED_NO_SOL"
        else:
            status = "PASS" if trial_result["agent_passed"] else "FAIL"
        print(f"  Result: {status} | score={trial_result['score']:.4f} | "
              f"steps={trial_result['steps_taken']} | time={trial_result['elapsed_seconds']}s")

    scores = [t["score"] for t in trials]
    passed = [t["agent_passed"] for t in trials]

    world_result = {
        "world_idx": world_idx,
        "difficulty": difficulty,
        "seed": seed,
        "trials": trials,
        "scores": scores,
        "avg_score": round(sum(scores) / len(scores), 4),
        "best_score": round(max(scores), 4),
        "pass_count": sum(passed),
        "pass_any": any(passed),
        "pass_all": all(passed),
    }

    print(f"\n  World {world_idx:02d} summary: avg_score={world_result['avg_score']:.4f}, "
          f"pass={sum(passed)}/{n_runs}, best={world_result['best_score']:.4f}")

    if trials and trials[0].get("optimal_cost"):
        _print_optimal_route({"optimal_pathway": {
            "target": "see task",
            "medicinal_value": 0,
            "base_toxicity": 0,
            "optimal_cost": trials[0]["optimal_cost"],
            "yield_g": None,
            "total_time_seconds": None,
            "purchase_cost": None,
            "process_cost": None,
            "purification_cost": None,
            "m1_scale_g": None,
            "steps": [],
        }})

    return world_result


def compute_statistics(world_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute pass@1, pass@3, pass^3, avg_score per difficulty and overall."""
    stats: Dict[str, Any] = {}

    for group_name, group_results in [("overall", world_results)] + [
        (d, [r for r in world_results if r["difficulty"] == d]) for d in DIFFICULTIES
    ]:
        if not group_results:
            continue
        n = len(group_results)
        all_scores = []
        for r in group_results:
            all_scores.extend(r["scores"])

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        pass_at_1 = sum(
            sum(t["agent_passed"] for t in r["trials"]) / len(r["trials"])
            for r in group_results
        ) / n

        pass_at_3 = sum(1 for r in group_results if r["pass_any"]) / n

        pass_pow_3 = sum(1 for r in group_results if r["pass_all"]) / n

        stats[group_name] = {
            "n_worlds": n,
            "n_trials": len(all_scores),
            "avg_score": round(avg_score, 4),
            "pass@1": round(pass_at_1, 4),
            "pass@3": round(pass_at_3, 4),
            "pass^3": round(pass_pow_3, 4),
        }

    return stats


def _save_checkpoint(config: AgentConfig, args, world_results: List[Dict[str, Any]], filepath: str) -> None:
    stats = compute_statistics(world_results) if world_results else {}
    output_data = {
        "config": {
            "model": config.model,
            "max_steps": config.max_steps,
            "n_runs": args.n_runs,
        },
        "statistics": stats,
        "world_results": world_results,
    }
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scientific research agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to agent config file")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--max-steps", type=int, default=120, help="Max agent steps per world")
    parser.add_argument("--memory-dir", type=str, default=None, help="Persistent memory directory")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--world-idx", type=int, default=None,
                        help="Run only a specific world index (0-39)")
    parser.add_argument("--difficulty", type=str, choices=DIFFICULTIES, default=None,
                        help="Run only worlds of this difficulty")
    parser.add_argument("--n-runs", type=int, default=RUNS_PER_WORLD,
                        help="Number of trials per world (default: 3)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a previous checkpoint/results JSON file (skip completed worlds)")
    parser.add_argument("--list-worlds", action="store_true",
                        help="List all 40 worlds and exit")
    args = parser.parse_args()

    if args.list_worlds:
        worlds = get_world_list()
        print(f"{'Idx':<5} {'Difficulty':<10} {'Seed':<8}")
        print("-" * 25)
        for w in worlds:
            print(f"{w['world_idx']:<5} {w['difficulty']:<10} {w['seed']:<8}")
        return

    if args.config:
        config = AgentConfig.from_file(args.config)
    else:
        config = AgentConfig()

    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key
    if args.base_url:
        config.base_url = args.base_url
    config.max_steps = args.max_steps
    if args.memory_dir:
        config.memory_dir = args.memory_dir
    config.verbose = not args.quiet

    log_level = logging.INFO if config.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    worlds = get_world_list()

    if args.world_idx is not None:
        if args.world_idx < 0 or args.world_idx >= len(worlds):
            print(f"Error: --world-idx must be 0-{len(worlds)-1}")
            sys.exit(1)
        worlds = [worlds[args.world_idx]]
    elif args.difficulty:
        worlds = [w for w in worlds if w["difficulty"] == args.difficulty]

    print(f"Evaluation Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Runs per world: {args.n_runs}")
    print(f"  Worlds to evaluate: {len(worlds)}")
    if args.world_idx is not None:
        print(f"  Single world: idx={args.world_idx}")
    elif args.difficulty:
        print(f"  Difficulty filter: {args.difficulty}")
    print()

    resumed_results: Dict[int, Dict[str, Any]] = {}
    if args.resume:
        if os.path.isfile(args.resume):
            with open(args.resume, "r") as f:
                prev_data = json.load(f)
            for wr in prev_data.get("world_results", []):
                resumed_results[wr["world_idx"]] = wr
            print(f"Resumed from: {args.resume} ({len(resumed_results)} worlds already completed)")
        else:
            print(f"Warning: resume file not found: {args.resume}, starting fresh.")

    print("Pre-sampling tasks...")
    tasks: Dict[int, Dict[str, Any]] = {}
    for w in worlds:
        tasks[w["world_idx"]] = presample_task(w["difficulty"], w["seed"])
    print(f"  {len(tasks)} tasks sampled successfully.")

    output_file = args.output or f"eval_results_{int(time.time())}.json"
    checkpoint_file = output_file.rsplit(".", 1)[0] + ".checkpoint.json"

    world_results = []
    for wr_prev in resumed_results.values():
        if any(w["world_idx"] == wr_prev["world_idx"] for w in worlds):
            world_results.append(wr_prev)

    for w in worlds:
        if w["world_idx"] in resumed_results:
            print(f"\n  [Skipped] World {w['world_idx']:02d} (already completed in resume file)")
            continue
        task = tasks[w["world_idx"]]
        try:
            wr = evaluate_world(config, w, task, n_runs=args.n_runs)
        except (KeyboardInterrupt, SystemExit):
            print(f"\n  [Interrupted] Saving checkpoint with {len(world_results)} completed worlds...")
            _save_checkpoint(config, args, world_results, checkpoint_file)
            print(f"  Checkpoint saved to: {checkpoint_file}")
            print(f"  Resume with: --resume {checkpoint_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"  World {w['world_idx']:02d} failed with error: {e}")
            print(f"  [Error] World {w['world_idx']:02d} failed: {e}")
            print(f"  Saving checkpoint with {len(world_results)} completed worlds...")
            _save_checkpoint(config, args, world_results, checkpoint_file)
            print(f"  Checkpoint saved to: {checkpoint_file}")
            print(f"  Resume with: --resume {checkpoint_file}")
            sys.exit(1)
        world_results.append(wr)
        _save_checkpoint(config, args, world_results, checkpoint_file)

    stats = compute_statistics(world_results)

    print(f"\n{'='*70}")
    print("EVALUATION STATISTICS")
    print(f"{'='*70}")
    print(f"\n{'Group':<10} {'N':<5} {'avg_score':<12} {'pass@1':<10} {'pass@3':<10} {'pass^3':<10}")
    print("-" * 57)
    for group in ["easy", "medium", "overall"]:
        if group in stats:
            s = stats[group]
            print(f"{group:<10} {s['n_worlds']:<5} {s['avg_score']:<12.4f} "
                  f"{s['pass@1']:<10.4f} {s['pass@3']:<10.4f} {s['pass^3']:<10.4f}")

    output_data = {
        "config": {
            "model": config.model,
            "max_steps": config.max_steps,
            "n_runs": args.n_runs,
        },
        "statistics": stats,
        "world_results": world_results,
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_file}")

    if os.path.isfile(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"  Checkpoint file removed: {checkpoint_file}")


if __name__ == "__main__":
    main()

