[English](README.md) | 中文

# Sci-Agents：科学研究智能体框架

这是一个面向 `xenoverse.sci_research_env` 化学环境的 LLM 科学研究智能体。智能体使用
ReAct 推理循环、OpenAI 兼容的函数调用，以及可持久化的实验记忆和化学知识抽取。

## 仓库结构

```text
.
|-- run.py                 # 运行单次智能体会话
|-- eval.py                # 运行固定的 60 世界评测基准
|-- configs/
|   `-- default.json       # 默认 AgentConfig 配置
`-- sci_agent/
    |-- agent.py           # ReAct 推理循环
    |-- config.py          # 配置 dataclass 与配置文件加载
    |-- llm/               # OpenAI 兼容 LLM 客户端
    |-- memory/            # 工作记忆、情景记忆、语义记忆
    `-- tools/
        `-- env_adapter.py # Xenoverse 环境工具适配器
```

## 安装

```bash
pip install -r requirements.txt
```

项目依赖 `xenoverse`。可以将 Xenoverse 仓库放在本仓库旁边的 `../Xenoverse`，也可以将其安装为
Python 包，或在运行 `eval.py` 前设置 `XENOVERSE_ROOT`：

```bash
export XENOVERSE_ROOT=/path/to/Xenoverse
```

通过命令行参数或环境变量设置 API 凭据：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"  # 可选
```

## 运行单次会话

```bash
# 存在 configs/default.json 时会自动使用它
python run.py --seed 42 --complexity medium

# 覆盖模型和步数预算
python run.py --model gpt-4o --max-steps 80 --complexity hard

# 加载自定义 JSON 或 YAML 配置
python run.py --config configs/default.json

# 跨会话持久化记忆
python run.py --seed 42 --memory-dir ./memory_store
```

`run.py` 会启动一个采样任务，并打印执行步数、最佳提交得分/成本和记忆摘要。

## 评测基准

`eval.py` 运行本仓库固定的评测基准：

- 总共 60 个预采样世界。
- `easy`、`medium`、`hard` 各 20 个世界。
- 种子固定：`easy=1000..1019`，`medium=2000..2019`，`hard=3000..3019`。
- 默认每个世界运行 3 次 trial。
- 按难度和总体统计 `avg_score`、`pass@1`、`pass@3`、`pass^3`。

列出所有评测世界：

```bash
python eval.py --list-worlds
```

运行完整评测：

```bash
python eval.py --model gpt-4o --max-steps 120 --output results/eval.json
```

运行较小子集：

```bash
# 运行单个世界，索引范围 0-59
python eval.py --world-idx 7 --n-runs 1 --output results/world_07.json

# 运行某一难度的所有世界
python eval.py --difficulty medium --n-runs 3 --output results/medium.json
```

恢复中断的评测：

```bash
python eval.py --resume results/eval.checkpoint.json --output results/eval.json
```

`eval.py` 会在每个世界完成后写入 checkpoint。评测成功完成后，最终 JSON 写入 `--output`，并删除
checkpoint 文件。

### 评测计分

对于可解任务，智能体提交有效解即为通过。分数上限为 `1.0`，计算方式为：

```text
optimal_cost / agent_best_cost
```

如果智能体在可解任务中声明无解，分数为 `0.0`。

对于无解任务，只有正确声明无解才算通过。分数会奖励更低成本的探索：

```text
min(1.0, baseline_cost / total_experiment_cost)
```

无解任务的 baseline 为：easy `50.0`，medium `100.0`，hard `200.0`。

### 评测命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 加载智能体配置文件。 |
| `--model` | 覆盖 LLM 模型名。 |
| `--api-key` | 覆盖 API key。 |
| `--base-url` | 覆盖 OpenAI 兼容 API base URL。 |
| `--max-steps` | 每次 trial 的最大智能体步数。默认：`120`。 |
| `--memory-dir` | 持久化记忆目录。评测内部会禁用每次 trial 的记忆目录，以保持 trial 独立。 |
| `--output` | 最终 JSON 输出路径。默认：`eval_results_<timestamp>.json`。 |
| `--quiet` | 降低日志详细程度。 |
| `--world-idx` | 只运行单个世界索引，范围 `0` 到 `59`。 |
| `--difficulty` | 只运行 `easy`、`medium` 或 `hard` 世界。 |
| `--n-runs` | 每个世界的 trial 数。默认：`3`。 |
| `--resume` | 从 checkpoint/results JSON 恢复，并跳过已完成世界。 |
| `--list-worlds` | 打印世界索引、难度和种子后退出。 |

## 编程方式使用

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

## 配置项

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `gpt-4o` | LLM 模型名。 |
| `api_key` | `None` | API key，或环境变量 `OPENAI_API_KEY`。 |
| `base_url` | `None` | API endpoint，或环境变量 `OPENAI_BASE_URL`。 |
| `temperature` | `0.7` | 采样温度。 |
| `max_tokens` | `4096` | 单次 LLM 调用的最大输出 token 数。 |
| `max_steps` | `50` | `run.py` 的最大智能体步数；`eval.py` 默认使用 `120`。 |
| `max_retries` | `3` | LLM 调用重试次数。 |
| `memory_dir` | `./memory_store` | 记忆持久化目录。 |
| `working_memory_max_messages` | `80` | 工作记忆保留的最大消息数。 |
| `working_memory_max_tokens` | `32000` | 工作记忆近似 token 预算。 |
| `complexity_level` | `None` | 世界复杂度：`easy`、`medium` 或 `hard`。 |
| `seed` | `None` | 任务采样随机种子。 |
| `verbose` | `true` | 是否打印运行日志。 |
| `log_file` | `None` | 可选日志文件路径。 |

## 依赖

- `xenoverse`
- `openai>=1.0.0`
- `tiktoken>=0.5.0`
- `numpy>=1.24.0`
- `scipy>=1.10.0`
- `pyyaml>=6.0`
