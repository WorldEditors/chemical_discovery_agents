English | [中文](README.zh.md)

# Sci-Agents: Scientific Research Agent Framework

An LLM-based scientific research agent for autonomous exploration in the
`xenoverse.chemverse` chemistry environment. The agent uses a ReAct loop,
OpenAI-compatible function calling, and persistent memory for experiment history
and extracted chemical knowledge.

## Repository Layout

```text
.
|-- run.py                 # Run one agent session
|-- eval.py                # Run the fixed 40-world evaluation benchmark
|-- data_synth.py          # Generate memory-carrying multi-world sequences
|-- configs/
|   `-- llm.json           # LLM model, API credentials, and endpoint
`-- sci_agent/
    |-- agent.py           # ReAct reasoning loop
    |-- config.py          # Configuration dataclass and file loading
    |-- trace.py           # Model and environment JSONL trajectories
    |-- llm/               # OpenAI-compatible LLM client
    |-- memory/            # Working, episodic, and semantic memory
    `-- tools/
        `-- env_adapter.py # Adapter for the Xenoverse environment tools
```

## Installation

```bash
pip install -r requirements.txt
```

The project depends on `xenoverse`. Keep the Xenoverse repo next to this repo as
`../Xenoverse`, install it as a package, or set `XENOVERSE_ROOT` before running
`run.py`, `eval.py`, or `data_synth.py`:

```bash
export XENOVERSE_ROOT=/path/to/Xenoverse
```

Set the model and API connection in `configs/llm.json`. The OpenAI client
loads this file automatically; `base_url` is the API root, not the
`/responses` or `/chat/completions` endpoint:

```bash
{
  "model": "your-model",
  "api_key": "your-key",
  "base_url": "https://your-endpoint/v1"
}
```

## Run One Agent Session

```bash
# Use configs/llm.json automatically
python run.py --seed 42 --complexity medium

# Change the step budget (set the model in configs/llm.json)
python run.py --max-steps 80 --complexity hard

# Optionally load a separate AgentConfig JSON or YAML file of your own
python run.py --config /path/to/agent-config.json

# Persist memory across sessions
python run.py --seed 42 --memory-dir ./memory_store
```

`run.py` starts one sampled task and prints the steps taken, best submitted
score/cost, and memory summary.

### Trajectories and memory

Each run appends two JSONL files under ./trajectories by default. Set
--trajectory-dir DIR to choose another location:

- fm_traj.jsonl stores every model call's full messages, tools, request
  parameters, provider request payload (for the OpenAI client), and raw response
  or error.
- e2e_traj.jsonl stores the public session start, each agent/environment
  tool input and output (including automatic progress queries), and session end.
  Hidden prebuilt world data is not copied into this agent-visible trajectory.

Each record has a UTC timestamp, unique run_id, session ID, step, and optional
evaluation context. Files are append-only so successive sessions remain
distinguishable.

run.py reuses the selected memory directory across invocations. The agent is
told that the next session may be the same world or a different one, and that
specific prior chemical facts must be checked against current observations.

## Evaluation Benchmark

`eval.py` runs the fixed benchmark used by this repository:

- 40 pre-sampled worlds in total.
- 20 `easy` and 20 `medium` worlds.
- Seeds are deterministic: `easy=1000..1019`, `medium=2000..2019`.
- Each world runs 3 trials by default.
- Metrics are reported per difficulty and overall: `avg_score`, `pass@1`,
  `pass@3`, and `pass^3`.

List benchmark worlds:

```bash
python eval.py --list-worlds
```

Run the full benchmark:

```bash
python eval.py --max-steps 120 --output results/eval.json
```

Run a smaller subset:

```bash
# One world by index, 0-39
python eval.py --world-idx 7 --n-runs 1 --output results/world_07.json

# All worlds at one difficulty
python eval.py --difficulty medium --n-runs 3 --output results/medium.json
```

Resume an interrupted evaluation:

```bash
python eval.py --resume results/eval.checkpoint.json --output results/eval.json
```

`eval.py` writes a checkpoint after each completed world. If the run completes
successfully, the final JSON is written to `--output` and the checkpoint file is
removed.

Evaluation keeps trials independent by default. To test memory carryover across
trials and worlds, specify both options:

    python eval.py --carry-memory --memory-dir results/shared_memory --output results/eval.json

The default trajectory directory for this command is
results/eval_trajectories; use --trajectory-dir to override it. Memory
carryover changes the evaluation protocol, so compare its scores separately
from independent-run results. On resume, use the same memory directory.

### Evaluation Scoring

For solvable tasks, the agent passes when it submits a valid solution. The score
is capped at `1.0` and is computed as:

```text
optimal_cost / agent_best_cost
```

If the agent declares no solution for a solvable task, the score is `0.0`.

For unsolvable tasks, the agent passes only when it declares no solution. The
score rewards cheaper investigation:

```text
min(1.0, baseline_cost / total_experiment_cost)
```

The unsolvable baselines are `50.0` for easy and `100.0` for medium.

### Evaluation CLI

| Flag | Description |
| --- | --- |
| `--config` | Load an AgentConfig file; the model and API connection still come from `configs/llm.json`. |
| `--max-steps` | Max agent steps per trial. Default: `120`. |
| `--memory-dir` | Persistent memory directory; used across trials with `--carry-memory`. |
| `--carry-memory` | Reuse memory across trials and worlds (requires `--memory-dir`). |
| `--trajectory-dir` | Directory for the two JSONL traces. |
| `--output` | Final JSON output path. Default: `eval_results_<timestamp>.json`. |
| `--quiet` | Reduce logging verbosity. |
| `--world-idx` | Run only one world index, from `0` to `39`. |
| `--difficulty` | Run only `easy` or `medium` worlds. |
| `--n-runs` | Trials per world. Default: `3`. |
| `--resume` | Resume from a checkpoint/results JSON and skip completed worlds. |
| `--list-worlds` | Print world indices, difficulties, and seeds, then exit. |

## Programmatic Usage

```python
from sci_agent import AgentConfig, SciResearchAgent

config = AgentConfig(
    max_steps=40,
    complexity_level="hard",
    memory_dir="./memory_store",
)

agent = SciResearchAgent(config=config)
result = agent.run(seed=42)

print(result["best_score"])
print(result["steps_taken"])
print(result["memory_summary"])
```

## Configuration

`AgentConfig` controls the agent and its memory. The LLM client reads `model`,
`api_key`, `base_url`, `timeout`, and its retry count from `configs/llm.json`.
The `--config` flag on `run.py`, `eval.py`, and `data_synth.py` loads an
optional, separate AgentConfig JSON/YAML file.

| Parameter | Default | Description |
| --- | --- | --- |
| `temperature` | `0.7` | Sampling temperature. |
| `max_tokens` | `4096` | Maximum output tokens for one LLM call. |
| `max_steps` | `50` | Maximum agent steps for `run.py`; `eval.py` defaults to `120`. |
| `max_retries` | `3` | Currently unused by the OpenAI client; set its retry count in `configs/llm.json`. |
| `memory_dir` | `./memory_store` | Memory persistence directory. |
| `trajectory_dir` | `./trajectories` | Directory for `fm_traj.jsonl` and `e2e_traj.jsonl`. |
| `working_memory_max_messages` | `240` | Maximum messages retained in working memory. |
| `working_memory_max_tokens` | `96000` | Approximate working-memory token budget. |
| `complexity_level` | `None` | World complexity: `easy`, `medium`, or `hard`. |
| `seed` | `None` | Random seed for task sampling. |
| `verbose` | `true` | Print runtime logs. |
| `log_file` | `None` | Optional log file path. |

## Dependencies

- `xenoverse`
- `openai>=1.0.0`
- `tiktoken>=0.5.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `pyyaml>=6.0`

## Long-sequence data synthesis

`data_synth.py` generates independent, memory-carrying records. For example:

```bash
python data_synth.py --sequences 8 --workers 4 --world-pool-size 5 \
  --world-visits 10 --trials-per-world 2 --max-steps 120 \
  --output-dir data_synth_output
```

The three counts are **nominal centers**, not fixed values. By default, each
sequence samples 3–7 distinct worlds (`--world-pool-jitter 2`) and 8–12 world
visits (`--world-visits-jitter 2`, always at least the sampled pool size).
Each visit independently samples 1–3 trials (`--trials-jitter 1`). All draws
are deterministic for a given sequence seed. Set any jitter to `0` to fix that
count; use `--world-pool-jitter 0 --world-visits-jitter 0 --trials-jitter 0`
for all fixed counts.
The realized counts and per-visit trial counts are saved in `source_manifest.json`.

Each sequence samples its own pool of worlds, visits every world at least once,
then may revisit earlier worlds without placing the same world in consecutive
visits. Each visit runs its sampled number of independent trials. Experiment
state resets for each trial, while episodic and semantic memory persists for
all trials and visits **within that sequence**. Different sequences have
separate memory directories and can run in separate processes.

For each sequence, the nearest integer to 40% of distinct worlds has its name
announced on entry; the remaining worlds have no agent-visible world ID and
receive one of 32 different unnamed transition prompts. The private world
names, generation seeds, exact task payloads, reveal flags, and visit schedule
are stored separately from agent-visible trajectories.

```text
data_synth_output/record-000000/
  source_manifest.json
  events.jsonl
  fm_traj.jsonl
  e2e_traj.jsonl
  memory/episodic.json
  memory/semantic.json
  private/worlds.json
  private/world-000.json
  COMMITTED
```

The `record-*` layout, semantic `events.jsonl`, event IDs, monotonic timestamps,
and separate private annotations follow the source-record ideas in ProcDataSyn.
The events are a JSONL interchange format, not pre-tokenized binary atoms.
`COMMITTED` marks a complete record; failed records retain partial traces and
have a `FAILED` marker. Choose a new output directory for a new generation run;
existing record directories are never appended or overwritten.
