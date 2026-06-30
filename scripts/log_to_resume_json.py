#!/usr/bin/env python3
"""Parse a stdout log of a previous eval.py run and rebuild a partial
results JSON suitable for `eval.py --resume`.

Only world blocks that contain a `World XX summary:` line are considered
complete and emitted. Per-trial fields preserved: score, agent_passed,
steps_taken, elapsed_seconds. Fields not recoverable from stdout
(optimal_cost, agent_best_cost, total_experiment_cost, etc.) are omitted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

WORLD_HEADER_RE = re.compile(
    r"^World\s+(\d+)\s+\|\s+(\w+)\s+\|\s+seed=(\d+)(\s+\[UNSOLVABLE\])?\s*$"
)
TRIAL_HEADER_RE = re.compile(r"^\s*---\s*Trial\s+(\d+)/(\d+)\s*---\s*$")
RESULT_RE = re.compile(
    r"^\s*Result:\s+(PASS|FAIL|CORRECT_NO_SOL|MISSED_NO_SOL)\s*\|\s*"
    r"score=([0-9.]+)\s*\|\s*steps=(\d+)\s*\|\s*time=([0-9.]+)s\s*$"
)
SUMMARY_RE = re.compile(
    r"^\s*World\s+(\d+)\s+summary:\s*avg_score=([0-9.]+),\s*pass=(\d+)/(\d+),\s*best=([0-9.]+)"
)


def parse_log(log_path: str) -> List[Dict[str, Any]]:
    worlds: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    pending_trial_idx: Optional[int] = None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            m = WORLD_HEADER_RE.match(line.strip())
            if m:
                if current is not None and not current.get("_completed"):
                    pass
                current = {
                    "world_idx": int(m.group(1)),
                    "difficulty": m.group(2),
                    "seed": int(m.group(3)),
                    "is_solvable": m.group(4) is None,
                    "trials": [],
                    "_completed": False,
                }
                worlds.append(current)
                pending_trial_idx = None
                continue

            if current is None:
                continue

            mt = TRIAL_HEADER_RE.match(line)
            if mt:
                pending_trial_idx = int(mt.group(1)) - 1
                continue

            mr = RESULT_RE.match(line)
            if mr and pending_trial_idx is not None:
                status = mr.group(1)
                score = float(mr.group(2))
                steps = int(mr.group(3))
                elapsed = float(mr.group(4))
                if current["is_solvable"]:
                    agent_passed = status == "PASS"
                else:
                    agent_passed = status == "CORRECT_NO_SOL"
                trial = {
                    "trial_idx": pending_trial_idx,
                    "agent_passed": agent_passed,
                    "score": round(score, 4),
                    "steps_taken": steps,
                    "elapsed_seconds": round(elapsed, 2),
                }
                if not current["is_solvable"]:
                    trial["declared_no_solution"] = status == "CORRECT_NO_SOL"
                current["trials"].append(trial)
                pending_trial_idx = None
                continue

            ms = SUMMARY_RE.match(line)
            if ms and int(ms.group(1)) == current["world_idx"]:
                current["_completed"] = True
                continue

    completed = []
    for w in worlds:
        if not w.get("_completed"):
            continue
        trials = w["trials"]
        scores = [t["score"] for t in trials]
        passed = [t["agent_passed"] for t in trials]
        wr = {
            "world_idx": w["world_idx"],
            "difficulty": w["difficulty"],
            "seed": w["seed"],
            "trials": trials,
            "scores": scores,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "best_score": round(max(scores), 4) if scores else 0.0,
            "pass_count": sum(passed),
            "pass_any": any(passed),
            "pass_all": all(passed) if passed else False,
        }
        completed.append(wr)

    return completed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", help="Path to stdout log file")
    p.add_argument("-o", "--output", default="eval_resume_from_log.json",
                   help="Output JSON path (default: eval_resume_from_log.json)")
    p.add_argument("--model", default="gpt-5.5",
                   help="Model name to record in config (default: gpt-5.5)")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--n-runs", type=int, default=3)
    args = p.parse_args()

    world_results = parse_log(args.log)
    if not world_results:
        print("No completed worlds found in log.", file=sys.stderr)
        sys.exit(1)

    out = {
        "config": {
            "model": args.model,
            "max_steps": args.max_steps,
            "n_runs": args.n_runs,
        },
        "statistics": {},
        "world_results": world_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    print(f"Parsed {len(world_results)} completed worlds from {args.log}")
    print(f"Wrote {args.output}")
    by_diff: Dict[str, int] = {}
    for w in world_results:
        by_diff[w["difficulty"]] = by_diff.get(w["difficulty"], 0) + 1
    print(f"By difficulty: {by_diff}")
    idxs = sorted(w["world_idx"] for w in world_results)
    print(f"World idx range: {idxs[0]}..{idxs[-1]} ({len(idxs)} worlds)")
    full = set(range(idxs[0], idxs[-1] + 1))
    missing_in_range = sorted(full - set(idxs))
    if missing_in_range:
        print(f"Missing in range: {missing_in_range}")


if __name__ == "__main__":
    main()
