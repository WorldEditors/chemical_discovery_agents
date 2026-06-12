#!/usr/bin/env python3
"""Evaluation script: runs the agent against xenoverse.sci_research_env and reports results."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xenoverse.sci_research_env.environment.backend import SciResearchBackend
from sci_agent import SciResearchAgent, AgentConfig
from sci_agent.tools.env_adapter import EnvironmentToolAdapter

logger = logging.getLogger(__name__)


def evaluate_single(
    config: AgentConfig,
    seed: int,
    complexity_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Run agent on a single world and collect evaluation metrics."""
    backend = SciResearchBackend()
    adapter = EnvironmentToolAdapter(backend=backend)

    agent = SciResearchAgent(config=config, env_adapter=adapter)

    start_time = time.time()
    result = agent.run(
        seed=seed,
        complexity_level=complexity_level,
        close_on_finish=False,
    )
    elapsed = time.time() - start_time

    session_id = adapter.session_id

    eval_result: Dict[str, Any] = {
        "seed": seed,
        "complexity_level": complexity_level,
        "agent_best_score": result["best_score"],
        "steps_taken": result["steps_taken"],
        "elapsed_seconds": round(elapsed, 2),
    }

    if session_id:
        try:
            best_submission = backend.eval_get_best_submission(session_id)
            if best_submission:
                eval_result["best_submission_detail"] = best_submission
        except Exception:
            pass

        try:
            optimal = backend.eval_find_cheapest_medicinal_pathway(session_id, min_medicinal_value=3.0)
            if optimal:
                eval_result["optimal_pathway"] = {
                    "target": optimal.get("target_compound"),
                    "optimal_score": optimal.get("score"),
                    "optimal_cost": optimal.get("total_cost"),
                    "num_steps": optimal.get("num_steps"),
                }
        except Exception:
            pass

        adapter.close()

    if eval_result.get("agent_best_score") and eval_result.get("optimal_pathway", {}).get("optimal_score"):
        optimal_score = eval_result["optimal_pathway"]["optimal_score"]
        agent_score = eval_result["agent_best_score"]
        eval_result["score_ratio"] = round(agent_score / optimal_score, 4) if optimal_score > 0 else None

    eval_result["memory_summary"] = result.get("memory_summary", "")

    return eval_result


def evaluate_batch(
    config: AgentConfig,
    seeds: List[int],
    complexity_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation over multiple seeds and aggregate results."""
    results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(seeds)}] Evaluating seed={seed}, complexity={complexity_level}")
        print(f"{'='*60}")

        try:
            r = evaluate_single(config, seed=seed, complexity_level=complexity_level)
            results.append(r)
            print(f"  Agent score: {r['agent_best_score']}")
            print(f"  Steps: {r['steps_taken']}")
            print(f"  Time: {r['elapsed_seconds']}s")
            if r.get("score_ratio") is not None:
                print(f"  Score ratio (agent/optimal): {r['score_ratio']}")
        except Exception as e:
            logger.error(f"  Failed on seed={seed}: {e}")
            results.append({"seed": seed, "error": str(e)})

    successful = [r for r in results if "error" not in r and r.get("agent_best_score") is not None]
    summary: Dict[str, Any] = {
        "total_runs": len(seeds),
        "successful_runs": len(successful),
        "failed_runs": len(results) - len(successful),
    }

    if successful:
        scores = [r["agent_best_score"] for r in successful]
        steps = [r["steps_taken"] for r in successful]
        times = [r["elapsed_seconds"] for r in successful]
        ratios = [r["score_ratio"] for r in successful if r.get("score_ratio") is not None]

        summary["avg_score"] = round(sum(scores) / len(scores), 2)
        summary["max_score"] = round(max(scores), 2)
        summary["min_score"] = round(min(scores), 2)
        summary["avg_steps"] = round(sum(steps) / len(steps), 1)
        summary["avg_time_seconds"] = round(sum(times) / len(times), 1)
        if ratios:
            summary["avg_score_ratio"] = round(sum(ratios) / len(ratios), 4)
            summary["max_score_ratio"] = round(max(ratios), 4)

    return {
        "summary": summary,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the scientific research agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to agent config file")
    parser.add_argument("--model", type=str, default=None, help="LLM model name")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to evaluate")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (e.g. 0,1,2,3,4)")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of sequential seeds starting from --seed-start")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed for sequential evaluation")
    parser.add_argument("--complexity", type=str, choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--max-steps", type=int, default=50, help="Max agent steps per world")
    parser.add_argument("--memory-dir", type=str, default=None, help="Persistent memory directory")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

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
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.memory_dir:
        config.memory_dir = args.memory_dir
    config.verbose = args.verbose

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"Evaluation Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Complexity: {args.complexity}")
    print(f"  Seeds: {seeds}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Memory dir: {config.memory_dir}")

    if len(seeds) == 1:
        result = evaluate_single(config, seed=seeds[0], complexity_level=args.complexity)
        print(f"\n{'='*60}")
        print("Evaluation Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        output_data = result
    else:
        batch_result = evaluate_batch(config, seeds=seeds, complexity_level=args.complexity)
        print(f"\n{'='*60}")
        print("Evaluation Summary:")
        print(json.dumps(batch_result["summary"], indent=2, ensure_ascii=False))
        output_data = batch_result

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
