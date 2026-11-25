#!/usr/bin/env python3
"""Generate long, memory-carrying chemistry exploration records."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import heapq
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
_xenoverse_root = os.environ.get("XENOVERSE_ROOT", str(REPO_ROOT.parent / "Xenoverse"))
if os.path.isdir(_xenoverse_root):
    sys.path.insert(0, os.path.abspath(_xenoverse_root))

from xenoverse.chemverse.task_sampler import SciResearchTaskSampler
from sci_agent import AgentConfig, SciResearchAgent
from sci_agent.tools.env_adapter import EnvironmentToolAdapter

logger = logging.getLogger(__name__)

# Every hidden-name template carries the same required disclosure. None contains
# a world identity, seed, or hint that would identify the current world.
HIDDEN_WORLD_TEMPLATES = (
    "你进入了一个新的世界。先观察，再决定哪些旧经验值得验证。",
    "你进入了一个新的世界。这里的化学规律需要重新通过实验确认。",
    "你进入了一个新的世界。请从当前可见信息开始探索。",
    "你进入了一个新的世界。过去的反应路线只能作为待检验的假设。",
    "你进入了一个新的世界。先检查可用材料和工具。",
    "你进入了一个新的世界。不要预设旧世界的物质性质仍然成立。",
    "你进入了一个新的世界。根据本轮观察规划下一步实验。",
    "你进入了一个新的世界。请谨慎区分记忆与当前证据。",
    "你进入了一个新的世界。可以借鉴旧策略，但要验证具体反应。",
    "你进入了一个新的世界。未知条件需要靠实验逐步缩小范围。",
    "你进入了一个新的世界。请先建立当前世界的可靠事实。",
    "你进入了一个新的世界。用观察结果更新你的研究计划。",
    "你进入了一个新的世界。旧结论可能有用，也可能失效。",
    "你进入了一个新的世界。先探查当前实验目标和约束。",
    "你进入了一个新的世界。保持探索记录，避免无依据的推断。",
    "你进入了一个新的世界。可从低成本实验开始检验猜想。",
    "你进入了一个新的世界。先弄清哪些材料确实可获得。",
    "你进入了一个新的世界。请依据当前反馈调整反应条件。",
    "你进入了一个新的世界。记忆中的化合物名称不保证性质相同。",
    "你进入了一个新的世界。每次失败也是关于条件的证据。",
    "你进入了一个新的世界。优先识别当前世界的关键约束。",
    "你进入了一个新的世界。请从可观测事实推断合成路径。",
    "你进入了一个新的世界。先验证最有价值的旧经验。",
    "你进入了一个新的世界。不要将未观察到的反应当成事实。",
    "你进入了一个新的世界。用工具逐步建立物质和反应图谱。",
    "你进入了一个新的世界。注意记录有效条件与失败条件。",
    "你进入了一个新的世界。请重新评估已有假设的可信度。",
    "你进入了一个新的世界。目标是以实验发现可行药物合成路线。",
    "你进入了一个新的世界。先找出值得分析的化合物。",
    "你进入了一个新的世界。根据当前证据控制探索成本。",
    "你进入了一个新的世界。旧世界的成功并不证明这里也会成功。",
    "你进入了一个新的世界。请先确认环境状态再采取行动。",
)

NAMED_WORLD_TEMPLATES = (
    "你进入了世界“{name}”。请核对旧记忆与当前观察。",
    "现在切换到世界“{name}”。先确认这里的实验条件。",
    "本轮世界名为“{name}”。过往记录仍需结合当前证据。",
    "你已抵达世界“{name}”。请继续科学探索。",
    "当前世界是“{name}”。先检查目标、材料与工具。",
    "已进入“{name}”世界。请据观察判断哪些经验可复用。",
    "世界已切换为“{name}”。从新会话的状态开始实验。",
    "你现在位于世界“{name}”。谨慎验证任何跨世界猜想。",
)

_NAME_FIRST = (
    "Aster", "Lumen", "Orion", "Vela", "Cedar", "Nova", "Iris", "Ember",
    "Atlas", "Mica", "Helix", "Nimbus", "Quartz", "Solace", "Cobalt", "Lyra",
)
_NAME_LAST = (
    "Vale", "Reach", "Harbor", "Field", "Crown", "Delta", "Arc", "Haven",
    "Ridge", "Bloom", "Basin", "Grove", "Spire", "Point", "Gate", "Dawn",
)


def plan_sequence(seed: int, pool_size: int, visits: int) -> Dict[str, Any]:
    """Return a deterministic schedule with distinct adjacent worlds and revisits."""
    if pool_size < 2 or visits < pool_size:
        raise ValueError("world-pool-size must be >= 2 and world-visits >= world-pool-size")
    rng = random.Random(seed)
    reveal_count = round(pool_size * 0.4)
    revealed = set(rng.sample(range(pool_size), reveal_count))
    used_seeds = set()
    worlds = []
    for index in range(pool_size):
        world_seed = rng.randrange(1, 2**31)
        while world_seed in used_seeds:
            world_seed = rng.randrange(1, 2**31)
        used_seeds.add(world_seed)
        name = f"{_NAME_FIRST[index % len(_NAME_FIRST)]}-{_NAME_LAST[(index // len(_NAME_FIRST)) % len(_NAME_LAST)]}-{world_seed:08x}"
        worlds.append({"world_index": index, "name": name, "requested_seed": world_seed,
                       "revealed": index in revealed})
    schedule = list(range(pool_size))
    rng.shuffle(schedule)
    while len(schedule) < visits:
        choices = [i for i in range(pool_size) if i != schedule[-1]]
        schedule.append(rng.choice(choices))
    return {"worlds": worlds, "schedule": schedule}


def sample_sequence_counts(spec: Dict[str, Any]) -> Tuple[int, int]:
    """Sample feasible world and visit counts from nominal values and jitter."""
    rng = random.Random(spec["seed"] ^ 0xA11CE)
    pool_center = spec["world_pool_size"]
    visit_center = spec["world_visits"]
    pool_jitter = spec.get("world_pool_jitter", 0)
    visit_jitter = spec.get("world_visits_jitter", 0)
    pool_low = max(2, pool_center - pool_jitter)
    pool_high = min(pool_center + pool_jitter, visit_center + visit_jitter)
    if pool_low > pool_high:
        raise ValueError("world-pool-size and world-visits jitter ranges have no feasible pair")
    pool_size = rng.randint(pool_low, pool_high)
    visit_low = max(pool_size, visit_center - visit_jitter)
    visits = rng.randint(visit_low, visit_center + visit_jitter)
    return pool_size, visits


def sample_trial_count(spec: Dict[str, Any], rng: random.Random) -> int:
    center = spec["trials_per_world"]
    jitter = spec.get("trials_jitter", 0)
    return rng.randint(max(1, center - jitter), center + jitter)


def _announcement(world: Dict[str, Any], rng: random.Random) -> Tuple[str, int]:
    if world["revealed"]:
        index = rng.randrange(len(NAMED_WORLD_TEMPLATES))
        return NAMED_WORLD_TEMPLATES[index].format(name=world["name"]), index
    index = rng.randrange(len(HIDDEN_WORLD_TEMPLATES))
    return HIDDEN_WORLD_TEMPLATES[index], index


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_trace(path: Path, source: str) -> Iterator[Tuple[int, str, Dict[str, Any]]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            yield int(record["timestamp_monotonic_ns"]), source, record


def _write_events(record_dir: Path) -> int:
    """Merge the two ordered traces into one ProcDataSyn-style semantic stream."""
    streams = [
        _read_trace(record_dir / "fm_traj.jsonl", "fm"),
        _read_trace(record_dir / "e2e_traj.jsonl", "e2e"),
    ]
    first_ns: Optional[int] = None
    previous_us = 0
    index = 0
    with (record_dir / "events.jsonl").open("w", encoding="utf-8") as output:
        for timestamp_ns, source, record in heapq.merge(*streams, key=lambda item: item[0]):
            if first_ns is None:
                first_ns = timestamp_ns
            timestamp_us = (timestamp_ns - first_ns) // 1000
            if source == "fm":
                parts = (("input", "model_input", "input", "user"),
                         ("output", "model_output", "target", "assistant"))
            else:
                event = record["event"]
                parts = (("input", f"{event}_input", "target", "assistant"),
                         ("output", f"{event}_output", "input", "tool"))
                if event in {"world_transition", "session_start", "session_end", "session_close"}:
                    parts = (("input", f"{event}_input", "input", "user"),
                             ("output", f"{event}_output", "input", "tool"))
            for field, kind, supervision, role in parts:
                value = record.get(field)
                if value is None:
                    continue
                event_record = {
                    "event_id": f"e{index:09d}", "sequence_index": index,
                    "timestamp_us": timestamp_us,
                    "delta_t_us": 0 if index == 0 else timestamp_us - previous_us,
                    "kind": kind, "supervision": supervision,
                    "message": {"role": role, "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}]},
                    "metadata": {"source": source, "run_id": record["run_id"],
                                 "step": record.get("step"), "session_id": record.get("session_id")},
                }
                output.write(json.dumps(event_record, ensure_ascii=False) + "\n")
                previous_us = timestamp_us
                index += 1
    return index


def run_sequence(
    spec: Dict[str, Any],
    agent_factory: Optional[Callable[[AgentConfig, EnvironmentToolAdapter], SciResearchAgent]] = None,
    task_sampler: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate one isolated record. Factories allow offline integration tests."""
    record_id = f"record-{spec['sequence_index']:06d}"
    record_dir = Path(spec["output_dir"]) / record_id
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / "private").mkdir()
    pool_size, visit_count = sample_sequence_counts(spec)
    plan = plan_sequence(spec["seed"], pool_size, visit_count)
    rng = random.Random(spec["seed"] ^ 0xC0FFEE)
    trial_rng = random.Random(spec["seed"] ^ 0xBADC0DE)
    config = AgentConfig.from_file(spec["config"]) if spec.get("config") else AgentConfig()
    config.max_steps = spec["max_steps"]
    config.memory_dir = str(record_dir / "memory")
    config.trajectory_dir = str(record_dir)
    config.verbose = spec.get("verbose", False)
    sampler = task_sampler or SciResearchTaskSampler
    manifest = {
        "format": "chemverse-semantic-events-v1", "record_id": record_id,
        "producer": "chemical_discovery_agents.data_synth",
        "environment": {"name": "chemverse", "seed": spec["seed"]},
        "clock": {"canonical_unit": "microsecond", "domain": "process_monotonic"},
        "files": {"events": "events.jsonl", "fm_trace": "fm_traj.jsonl",
                  "e2e_trace": "e2e_traj.jsonl", "private_annotations": "private/worlds.json"},
        "world_pool_size": pool_size,
        "world_visits": visit_count,
        "requested_counts": {
            "world_pool_size": spec["world_pool_size"],
            "world_pool_jitter": spec.get("world_pool_jitter", 0),
            "world_visits": spec["world_visits"],
            "world_visits_jitter": spec.get("world_visits_jitter", 0),
            "trials_per_world": spec["trials_per_world"],
            "trials_jitter": spec.get("trials_jitter", 0),
        },
        "status": "in_progress",
    }
    _write_json(record_dir / "source_manifest.json", manifest)
    worlds = plan["worlds"]
    private = {"worlds": worlds, "schedule": plan["schedule"], "visits": []}
    try:
        tasks = []
        for world in worlds:
            task = sampler(seed=world["requested_seed"],
                           complexity_level=spec["difficulty"], world_id=world["name"])
            world["actual_seed"] = task["seed"]
            task_file = f"world-{world['world_index']:03d}.json"
            _write_json(record_dir / "private" / task_file, task)
            world["task_file"] = task_file
            tasks.append(task)
        _write_json(record_dir / "private" / "worlds.json", private)

        for visit_index, world_index in enumerate(plan["schedule"]):
            world = worlds[world_index]
            intro, template_index = _announcement(world, rng)
            planned_trials = sample_trial_count(spec, trial_rng)
            visit = {"visit_index": visit_index, "world_index": world_index,
                     "template_index": template_index,
                     "planned_trials": planned_trials, "trials": []}
            private["visits"].append(visit)
            for trial_index in range(planned_trials):
                notice = intro if trial_index == 0 else (
                    "这是同一个世界中的下一次独立实验。实验环境已重置，"
                    "请结合此前记忆，并重新检查当前状态。"
                )
                adapter = EnvironmentToolAdapter()
                adapter.hide_world_id = not world["revealed"]
                agent = (agent_factory(config, adapter) if agent_factory
                         else SciResearchAgent(config=config, env_adapter=adapter))
                try:
                    result = agent.run(
                        task=tasks[world_index], max_steps=spec["max_steps"],
                        session_intro=notice,
                        trace_context={"record_id": record_id, "visit_index": visit_index,
                                       "trial_index": trial_index},
                    )
                    visit["trials"].append({"trial_index": trial_index,
                                            "steps_taken": result["steps_taken"],
                                            "best_score": result["best_score"]})
                finally:
                    if adapter.session_id:
                        adapter.close()
                _write_json(record_dir / "private" / "worlds.json", private)
        event_count = _write_events(record_dir)
        manifest["status"] = "complete"
        manifest["event_count"] = event_count
        manifest["trial_count"] = sum(len(v["trials"]) for v in private["visits"])
        manifest["trials_per_visit"] = [v["planned_trials"] for v in private["visits"]]
        _write_json(record_dir / "source_manifest.json", manifest)
        (record_dir / "COMMITTED").touch()
        return {"record_id": record_id, "status": "complete", "event_count": event_count,
                "world_pool_size": pool_size, "world_visits": visit_count,
                "trials_per_visit": manifest["trials_per_visit"],
                "trial_count": manifest["trial_count"]}
    except Exception as exc:
        _write_json(record_dir / "private" / "worlds.json", private)
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(record_dir / "source_manifest.json", manifest)
        (record_dir / "FAILED").touch()
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data_synth_output")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--world-pool-size", type=int, default=5)
    parser.add_argument("--world-pool-jitter", type=int, default=2,
                        help="Random +/- variation around world-pool-size (default: 2)")
    parser.add_argument("--world-visits", type=int, default=10)
    parser.add_argument("--world-visits-jitter", type=int, default=2,
                        help="Random +/- variation around world-visits (default: 2)")
    parser.add_argument("--trials-per-world", type=int, default=2,
                        help="Number of independent trials per world visit")
    parser.add_argument("--trials-jitter", type=int, default=1,
                        help="Random +/- variation per world visit (default: 1)")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--config", default=None, help="AgentConfig JSON/YAML file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.sequences < 1 or args.workers < 1 or args.trials_per_world < 1 or args.max_steps < 1:
        parser.error("sequences, workers, trials-per-world, and max-steps must be >= 1")
    if args.world_pool_size < 2 or args.world_visits < 1:
        parser.error("world-pool-size must be >= 2 and world-visits >= 1")
    if min(args.world_pool_jitter, args.world_visits_jitter, args.trials_jitter) < 0:
        parser.error("jitter values must be nonnegative")
    if args.world_visits + args.world_visits_jitter < max(2, args.world_pool_size - args.world_pool_jitter):
        parser.error("world-pool-size and world-visits jitter ranges have no feasible pair")
    output_dir = str(Path(args.output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    specs = [{**vars(args), "output_dir": output_dir,
              "sequence_index": i, "seed": args.seed + i * 1000003}
             for i in range(args.sequences)]
    failures = 0
    if args.workers == 1:
        for spec in specs:
            try:
                result = run_sequence(spec)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception:
                logger.exception("Sequence %s failed", spec["sequence_index"])
                failures += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_sequence, spec): spec for spec in specs}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result(), ensure_ascii=False), flush=True)
                except Exception:
                    logger.exception("Sequence %s failed", futures[future]["sequence_index"])
                    failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
