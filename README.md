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
|-- configs/
|   `-- default.json       # Default AgentConfig values
`-- sci_agent/
    |-- agent.py           # ReAct reasoning loop
    |-- config.py          # Configuration dataclass and file loading
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
`eval.py`:

```bash
export XENOVERSE_ROOT=/path/to/Xenoverse
```

Set API credentials through CLI flags or environment variables:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"  # optional
```

## Run One Agent Session

```bash
# Use configs/default.json when present
python run.py --seed 42 --complexity medium

# Override model and step budget
python run.py --model gpt-4o --max-steps 80 --complexity hard

# Load a custom JSON or YAML config
python run.py --config configs/default.json

# Persist memory across sessions
python run.py --seed 42 --memory-dir ./memory_store
```

`run.py` starts one sampled task and prints the steps taken, best submitted
score/cost, and memory summary.

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
python eval.py --model gpt-4o --max-steps 120 --output results/eval.json
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
| `--config` | Load an agent config file. |
| `--model` | Override the LLM model name. |
| `--api-key` | Override the API key. |
| `--base-url` | Override the OpenAI-compatible API base URL. |
| `--max-steps` | Max agent steps per trial. Default: `120`. |
| `--memory-dir` | Persistent memory directory. Evaluation disables memory per trial internally to keep trials independent. |
| `--output` | Final JSON output path. Default: `eval_results_<timestamp>.json`. |
| `--quiet` | Reduce logging verbosity. |
| `--world-idx` | Run only one world index, from `0` to `59`. |
| `--difficulty` | Run only `easy` or `medium` worlds. |
| `--n-runs` | Trials per world. Default: `3`. |
| `--resume` | Resume from a checkpoint/results JSON and skip completed worlds. |
| `--list-worlds` | Print world indices, difficulties, and seeds, then exit. |

## Programmatic Usage

```python
from sci_agent import AgentConfig, SciResearchAgent

config = AgentConfig(
    model="gpt-4o",
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

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | `gpt-4o` | LLM model name. |
| `api_key` | `None` | API key, or `OPENAI_API_KEY` from the environment. |
| `base_url` | `None` | API endpoint, or `OPENAI_BASE_URL` from the environment. |
| `temperature` | `0.7` | Sampling temperature. |
| `max_tokens` | `4096` | Maximum output tokens for one LLM call. |
| `max_steps` | `50` | Maximum agent steps for `run.py`; `eval.py` defaults to `120`. |
| `max_retries` | `3` | LLM call retry count. |
| `memory_dir` | `./memory_store` | Memory persistence directory. |
| `working_memory_max_messages` | `80` | Maximum messages retained in working memory. |
| `working_memory_max_tokens` | `32000` | Approximate working-memory token budget. |
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
