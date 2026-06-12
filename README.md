# Sci-Agents: Scientific Research Agent Framework

An LLM-based scientific research agent framework for autonomous exploration and knowledge accumulation in a procedurally generated chemistry world, built around a ReAct reasoning loop and long-term memory management.

## Architecture

```text
sci_agent/
├── agent.py              # ReAct reasoning loop (Think → Act → Observe)
├── config.py             # Configuration management
├── llm/
│   ├── base.py           # Abstract LLM client interface
│   └── openai_client.py  # OpenAI-compatible implementation
├── memory/
│   ├── working.py        # Working memory (context window management)
│   ├── episodic.py       # Episodic memory (persisted experiment history)
│   ├── semantic.py       # Semantic memory (structured knowledge base)
│   └── manager.py        # Unified memory coordinator
└── tools/
    └── env_adapter.py    # Environment interaction adapter
```

## Core Design

### ReAct Agent Loop

The agent follows the ReAct (Reasoning + Acting) pattern:

1. **Think** - The LLM receives the current context (task description, memory, and observation history), reasons about it, and decides the next action.
2. **Act** - The LLM's function call is dispatched into the environment for execution.
3. **Observe** - The environment response is recorded into memory and fed back to the LLM.

At each step, the LLM's reasoning (`response.content`) and tool calls are produced together. The reasoning trace is retained in both working memory and episodic memory.

### Three-Tier Memory System

| Layer | Purpose | Persistent | Cross-Session |
|------|------|--------|-----------|
| **Working Memory** | Manages conversation history inside the LLM context window and handles truncation | No | No |
| **Episodic Memory** | Records action/observation/reasoning for each experiment | Yes (JSON) | Yes |
| **Semantic Memory** | Structured knowledge graph of compound properties, reaction relationships, and strategy insights | Yes (JSON) | Yes |

**Memory lifecycle:**
- After every tool call, semantic information such as compound properties and reaction products is automatically extracted into semantic memory.
- At the end of a session, an `ExperimentSummary` is generated and stored in episodic memory.
- When a new session starts, persisted memory is assembled into context and injected into the system prompt.

### Automatic Knowledge Extraction

`MemoryManager._extract_semantic()` automatically extracts structured knowledge from tool results:
- `analyze_compound` -> updates compound properties and medicinal-value clues
- `perform_reaction` -> records reactant/product relationships and experiment conditions
- `estimate_cost` -> accumulates cost-model insights
- `submit_solution` -> records the best score and successful strategies

## Quick Start

### Installation

```bash
cd sci_agents
pip install -r requirements.txt
```

Make sure the `xenoverse` package is importable, or keep `environment/` and `world_gen/` in the current directory.

### Run

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"  # optional, works with any OpenAI-compatible API

# Run with the default configuration
python run.py --seed 42 --complexity medium

# Specify model and step count
python run.py --model gpt-4o --max-steps 30 --complexity hard

# Use a custom config file
python run.py --config configs/default.json

# Use a custom memory directory for cross-session learning
python run.py --seed 42 --memory-dir ./my_memory
```

### Programmatic Usage

```python
from sci_agent import SciResearchAgent, AgentConfig

config = AgentConfig(
    model="gpt-4o",
    max_steps=40,
    complexity_level="hard",
    memory_dir="./memory_store",
)

agent = SciResearchAgent(config=config)
result = agent.run(seed=42)

print(f"Best score: {result['best_score']}")
print(f"Steps: {result['steps_taken']}")
print(f"Knowledge:\n{result['memory_summary']}")
```

### Cross-Session Experience Accumulation

```python
# Session 1: explore world seed=42
agent = SciResearchAgent(config=AgentConfig(memory_dir="./shared_memory"))
agent.run(seed=42)

# Session 2: explore a new world while reusing prior strategy insights
agent = SciResearchAgent(config=AgentConfig(memory_dir="./shared_memory"))
agent.run(seed=100)
# The system prompt will automatically include discoveries and strategies from earlier sessions
```

## Configuration

| Parameter | Default | Description |
|------|--------|------|
| `model` | `gpt-4o` | LLM model name |
| `api_key` | `None` | API key, or `OPENAI_API_KEY` from the environment |
| `base_url` | `None` | API endpoint, or `OPENAI_BASE_URL` from the environment |
| `temperature` | `0.7` | Sampling temperature |
| `max_tokens` | `4096` | Maximum output tokens for a single LLM call |
| `max_steps` | `50` | Maximum reasoning steps |
| `memory_dir` | `./memory_store` | Memory persistence directory |
| `complexity_level` | `None` | World complexity (`easy` / `medium` / `hard`) |
| `seed` | `None` | Random seed for world generation |
| `verbose` | `true` | Whether to print runtime logs |

## Dependencies

- `openai>=1.0.0` - LLM API client
- `tiktoken>=0.5.0` - Token counting
- `numpy>=1.24.0` - Environment dependency
- `scipy>=1.10.0` - Environment dependency
- `pyyaml>=6.0` - YAML config support (optional)
