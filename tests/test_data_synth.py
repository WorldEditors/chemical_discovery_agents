"""Offline checks for long sequence synthesis; no API calls are made."""
import json
from pathlib import Path
import random
import tempfile
import unittest

from data_synth import (HIDDEN_WORLD_TEMPLATES, plan_sequence, run_sequence,
                        sample_sequence_counts, sample_trial_count)
from sci_agent.agent import SciResearchAgent
from sci_agent.llm.base import BaseLLMClient, LLMResponse


class FinishLLM(BaseLLMClient):
    model = "offline-test"

    def __init__(self, prompts=None):
        self.prompts = prompts

    def chat(self, messages, **kwargs):
        if self.prompts is not None:
            self.prompts.append(messages[0]["content"])
        return LLMResponse(
            content="Finish", raw={"test": True},
            tool_calls=[{"id": "test-call", "name": "finish_experiment", "arguments": {}}],
        )

    def count_tokens(self, messages):
        return 0


class DataSynthTests(unittest.TestCase):
    def test_schedule_and_templates(self):
        plan = plan_sequence(123, 5, 10)
        self.assertEqual(sum(world["revealed"] for world in plan["worlds"]), 2)
        self.assertEqual(set(plan["schedule"]), set(range(5)))
        self.assertTrue(all(a != b for a, b in zip(plan["schedule"], plan["schedule"][1:])))
        self.assertGreaterEqual(len(set(HIDDEN_WORLD_TEMPLATES)), 30)
        self.assertTrue(all(text.startswith("你进入了一个新的世界") for text in HIDDEN_WORLD_TEMPLATES))

    def test_seeded_count_variation(self):
        counts = []
        trial_counts = []
        for seed in range(100, 150):
            spec = {"seed": seed, "world_pool_size": 5, "world_pool_jitter": 2,
                    "world_visits": 10, "world_visits_jitter": 2,
                    "trials_per_world": 2, "trials_jitter": 1}
            pool_size, visits = sample_sequence_counts(spec)
            self.assertEqual((pool_size, visits), sample_sequence_counts(spec))
            self.assertTrue(3 <= pool_size <= 7)
            self.assertTrue(8 <= visits <= 12)
            self.assertGreaterEqual(visits, pool_size)
            count = sample_trial_count(spec, random.Random(seed))
            self.assertTrue(1 <= count <= 3)
            counts.append((pool_size, visits))
            trial_counts.append(count)
        self.assertGreater(len({pool for pool, _ in counts}), 1)
        self.assertGreater(len({visits for _, visits in counts}), 1)
        self.assertEqual(set(trial_counts), {1, 2, 3})

    def test_record_memory_and_hidden_name(self):
        prompts = []

        def factory(config, adapter):
            return SciResearchAgent(config=config, llm_client=FinishLLM(prompts), env_adapter=adapter)

        with tempfile.TemporaryDirectory() as output_dir:
            result = run_sequence({
                "sequence_index": 0, "output_dir": output_dir, "seed": 314159,
                "world_pool_size": 2, "world_visits": 3, "trials_per_world": 2,
                "difficulty": "easy", "max_steps": 1, "verbose": False,
            }, agent_factory=factory)
            record = Path(output_dir) / "record-000000"
            private = json.loads((record / "private/worlds.json").read_text())
            hidden = next(world for world in private["worlds"] if not world["revealed"])
            traces = "".join((record / name).read_text() for name in (
                "fm_traj.jsonl", "e2e_traj.jsonl", "events.jsonl",
            ))
            self.assertNotIn(hidden["name"], traces)
            self.assertEqual(result["trial_count"], 6)
            self.assertTrue((record / "COMMITTED").exists())
            self.assertEqual(len(prompts), 6)
            self.assertIn("Past Session Insights", prompts[1])
            fm = [json.loads(line) for line in (record / "fm_traj.jsonl").read_text().splitlines()]
            self.assertEqual(len({line["run_id"] for line in fm}), 6)
            events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
            self.assertEqual([e["sequence_index"] for e in events], list(range(len(events))))
            self.assertTrue(all(e["delta_t_us"] >= 0 for e in events))


if __name__ == "__main__":
    unittest.main()
