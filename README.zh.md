[English](README.md) | 中文

# Sci-Agents：科学研究智能体框架

这是一个面向 `xenoverse.chemverse` 化学环境的 LLM 科学研究智能体。智能体使用
ReAct 推理循环、OpenAI 兼容的函数调用，以及可持久化的实验记忆和化学知识抽取。

## 仓库结构

```text
.
|-- run.py                 # 运行单次智能体会话
|-- eval.py                # 运行固定的 40 世界评测基准
|-- data_synth.py          # 生成跨世界、延续记忆的长序列
|-- configs/
|   `-- llm.json           # LLM 模型、API 凭据与服务地址
`-- sci_agent/
    |-- agent.py           # ReAct 推理循环
    |-- config.py          # 配置 dataclass 与配置文件加载
    |-- trace.py           # 模型和环境的 JSONL 轨迹
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
Python 包，或在运行 `run.py`、`eval.py`、`data_synth.py` 前设置 `XENOVERSE_ROOT`：

```bash
export XENOVERSE_ROOT=/path/to/Xenoverse
```

在 `configs/llm.json` 中设置模型和 API 连接信息。OpenAI 客户端会自动读取该文件；
`base_url` 应是 API 根路径，不包含 `/responses` 或 `/chat/completions`：

```bash
{
  "model": "your-model",
  "api_key": "your-key",
  "base_url": "https://your-endpoint/v1"
}
```

## 运行单次会话

```bash
# 自动读取 configs/llm.json
python run.py --seed 42 --complexity medium

# 修改步数预算（模型在 configs/llm.json 中设置）
python run.py --max-steps 80 --complexity hard

# 可选：加载自行创建的 AgentConfig JSON 或 YAML 文件
python run.py --config /path/to/agent-config.json

# 跨会话持久化记忆
python run.py --seed 42 --memory-dir ./memory_store
```

`run.py` 会启动一个采样任务，并打印执行步数、最佳提交得分/成本和记忆摘要。

### 轨迹与跨 Session 记忆

默认在 ./trajectories 追加写入两个 JSONL 文件；可用
--trajectory-dir DIR 指定目录：

- fm_traj.jsonl：每次模型调用的完整消息、工具定义、请求参数、
  OpenAI 客户端实际发送的请求体，以及原始响应或错误。
- e2e_traj.jsonl：公开的 Session 初始信息、智能体与环境之间每次工具调用的
  输入输出（包括自动进度查询）和结束事件；不会写入预生成任务中隐藏的世界真值。

每条记录包含 UTC 时间、唯一 run_id、Session ID 和步骤号。run.py
会跨调用复用指定的记忆目录。提示词会告知智能体下一轮可能是同一个世界，
也可能是不同世界；旧的化学事实需要根据当前观察重新验证。

## 评测基准

`eval.py` 运行本仓库固定的评测基准：

- 总共 40 个预采样世界。
- `easy`、`medium` 各 20 个世界。
- 种子固定：`easy=1000..1019`，`medium=2000..2019`。
- 默认每个世界运行 3 次 trial。
- 按难度和总体统计 `avg_score`、`pass@1`、`pass@3`、`pass^3`。

列出所有评测世界：

```bash
python eval.py --list-worlds
```

运行完整评测：

```bash
python eval.py --max-steps 120 --output results/eval.json
```

运行较小子集：

```bash
# 运行单个世界，索引范围 0-39
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

评测默认保持各次 trial 独立。若要让记忆在 trial 与世界之间延续，使用：

    python eval.py --carry-memory --memory-dir results/shared_memory --output results/eval.json

该命令默认将轨迹写入 results/eval_trajectories，也可用
--trajectory-dir 覆盖。开启记忆延续会改变评测协议，结果应与独立
trial 分开比较；恢复评测时应继续使用同一个记忆目录。

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

无解任务的 baseline 为：easy `50.0`，medium `100.0`。

### 评测命令行参数

| 参数 | 说明 |
| --- | --- |
| `--config` | 加载 AgentConfig 文件；模型与 API 连接仍从 `configs/llm.json` 读取。 |
| `--max-steps` | 每次 trial 的最大智能体步数。默认：`120`。 |
| `--memory-dir` | 持久化记忆目录；与 `--carry-memory` 一起使用时跨 trial 共享。 |
| `--carry-memory` | 跨 trial 和世界延续记忆（须同时设置 `--memory-dir`）。 |
| `--trajectory-dir` | 两份 JSONL 轨迹的保存目录。 |
| `--output` | 最终 JSON 输出路径。默认：`eval_results_<timestamp>.json`。 |
| `--quiet` | 降低日志详细程度。 |
| `--world-idx` | 只运行单个世界索引，范围 `0` 到 `39`。 |
| `--difficulty` | 只运行 `easy` 或 `medium` 世界。 |
| `--n-runs` | 每个世界的 trial 数。默认：`3`。 |
| `--resume` | 从 checkpoint/results JSON 恢复，并跳过已完成世界。 |
| `--list-worlds` | 打印世界索引、难度和种子后退出。 |

## 编程方式使用

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

## 配置项

`AgentConfig` 用于控制智能体及其记忆。LLM 客户端从 `configs/llm.json`
读取 `model`、`api_key`、`base_url`、`timeout` 和重试次数。
`run.py`、`eval.py`、`data_synth.py` 的
`--config` 用来加载另一份可选的 AgentConfig JSON/YAML 文件。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `temperature` | `0.7` | 采样温度。 |
| `max_tokens` | `4096` | 单次 LLM 调用的最大输出 token 数。 |
| `max_steps` | `50` | `run.py` 的最大智能体步数；`eval.py` 默认使用 `120`。 |
| `max_retries` | `3` | 该字段目前未接入 OpenAI 客户端；其重试次数在 `configs/llm.json` 中设置。 |
| `memory_dir` | `./memory_store` | 记忆持久化目录。 |
| `trajectory_dir` | `./trajectories` | `fm_traj.jsonl` 和 `e2e_traj.jsonl` 的保存目录。 |
| `working_memory_max_messages` | `240` | 工作记忆保留的最大消息数。 |
| `working_memory_max_tokens` | `96000` | 工作记忆近似 token 预算。 |
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

## 长序列数据合成

`data_synth.py` 可并行生成多条彼此独立、在序列内部延续记忆的长序列：

```bash
python data_synth.py --sequences 8 --workers 4 --world-pool-size 5 \
  --world-visits 10 --trials-per-world 2 --max-steps 120 \
  --output-dir data_synth_output
```

这三个数现在是**随机采样的中心值**。默认每条序列抽取 3–7 个不同世界
（`--world-pool-jitter 2`）和 8–12 次世界访问（`--world-visits-jitter 2`，
访问次数始终不少于实际世界数）。每次访问独立抽取 1–3 个 trial
（`--trials-jitter 1`）。给定序列 seed 后结果可复现。若要固定数量，
同时传入 `--world-pool-jitter 0 --world-visits-jitter 0 --trials-jitter 0`。
实际抽取数量和每次访问的 trial 数写入 `source_manifest.json`。

每条序列有自己的世界池，先进入所有世界至少一次，之后可回到先前世界；
相邻两次访问不会是同一世界。每次访问运行当次抽取数量的独立 trial。
实验环境在每个 trial 重置，但情景记忆和语义记忆在同一序列内持续保留；
不同序列的记忆和轨迹分开，可由不同进程并行生成。

按不同世界数量计算，最接近 40% 的世界在切换进入时公开名字。
其余世界不向 agent 透露世界 ID，并从 32 个匿名切换提示模板中选择。
真实名称、种子、完整任务、是否公开及访问顺序放在独立的私有标注中。

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

目录布局、语义事件、单调时间戳和私有标注参考 ProcDataSyn 的 source record
思路。`events.jsonl` 是尚未 token 化的交换格式，不是二进制 atom 文件。
只有出现 `COMMITTED` 才表示记录完整；失败记录保留部分轨迹并标记 `FAILED`。
再次生成请使用新的输出目录，工具不会覆盖已有的 record 目录。
