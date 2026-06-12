# Sci-Agents: Scientific Research Agent Framework

基于 LLM 的科学研究智能体框架，通过 ReAct 推理循环与长期记忆管理，在程序化生成的化学世界中进行自主探索和知识积累。

## 架构

```
sci_agent/
├── agent.py              # ReAct 推理循环 (Think → Act → Observe)
├── config.py             # 配置管理
├── llm/
│   ├── base.py           # LLM 客户端抽象接口
│   └── openai_client.py  # OpenAI 兼容实现
├── memory/
│   ├── working.py        # 工作记忆 (上下文窗口管理)
│   ├── episodic.py       # 情景记忆 (实验经历持久化)
│   ├── semantic.py       # 语义记忆 (结构化知识库)
│   └── manager.py        # 统一记忆协调器
└── tools/
    └── env_adapter.py    # 环境交互适配器
```

## 核心设计

### ReAct Agent Loop

智能体采用 ReAct (Reasoning + Acting) 范式运行：

1. **Think** — LLM 接收当前上下文（任务描述 + 记忆 + 观察历史），进行推理并决定下一步行动
2. **Act** — 将 LLM 的 function call 分发到环境中执行
3. **Observe** — 接收环境返回的结果，记录到记忆系统，反馈给 LLM

每一步中，LLM 的 reasoning（`response.content`）与 tool calls 同时产生，reasoning 作为思考链被保留在工作记忆和情景记忆中。

### 三级记忆系统

| 层级 | 作用 | 持久化 | 跨 Session |
|------|------|--------|-----------|
| **Working Memory** | 管理 LLM 上下文窗口内的对话历史，处理截断 | 否 | 否 |
| **Episodic Memory** | 记录每次实验的 action/observation/reasoning | 是 (JSON) | 是 |
| **Semantic Memory** | 结构化知识图谱：化合物属性、反应关系、策略洞察 | 是 (JSON) | 是 |

**记忆生命周期：**
- 每次 tool call 执行后，自动提取语义信息（化合物属性、反应产物等）写入 Semantic Memory
- Session 结束时，生成 ExperimentSummary 存入 Episodic Memory
- 新 Session 启动时，从持久化记忆中组装上下文注入 system prompt

### 知识自动提取

`MemoryManager._extract_semantic()` 从工具调用结果中自动提取结构化知识：
- `analyze_compound` → 更新化合物属性、药用价值线索
- `perform_reaction` → 记录反应物/产物关系、实验条件
- `estimate_cost` → 积累成本模型洞察
- `submit_solution` → 记录最佳得分和成功策略

## 快速开始

### 安装

```bash
cd sci_agents
pip install -r requirements.txt
```

确保 `xenoverse` 包可导入（或将 `environment/`、`world_gen/` 保留在当前目录）。

### 运行

```bash
# 设置 API Key
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-endpoint/v1"  # 可选，兼容任何 OpenAI API 接口

# 使用默认配置运行
python run.py --seed 42 --complexity medium

# 指定模型和步数
python run.py --model gpt-4o --max-steps 30 --complexity hard

# 使用自定义配置文件
python run.py --config configs/default.json

# 指定记忆存储目录（跨 session 积累经验）
python run.py --seed 42 --memory-dir ./my_memory
```

### 编程方式使用

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

### 跨 Session 经验积累

```python
# Session 1: 探索 world seed=42
agent = SciResearchAgent(config=AgentConfig(memory_dir="./shared_memory"))
agent.run(seed=42)

# Session 2: 探索新世界，但利用之前积累的策略洞察
agent = SciResearchAgent(config=AgentConfig(memory_dir="./shared_memory"))
agent.run(seed=100)
# system prompt 中会自动注入过去 session 的发现和策略
```

## 配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `gpt-4o` | LLM 模型名称 |
| `api_key` | `None` | API Key（或环境变量 `OPENAI_API_KEY`） |
| `base_url` | `None` | API 地址（或环境变量 `OPENAI_BASE_URL`） |
| `temperature` | `0.7` | 采样温度 |
| `max_tokens` | `4096` | 单次 LLM 最大输出 token |
| `max_steps` | `50` | 最大推理步数 |
| `memory_dir` | `./memory_store` | 记忆持久化目录 |
| `complexity_level` | `None` | 世界复杂度 (easy/medium/hard) |
| `seed` | `None` | 世界生成随机种子 |
| `verbose` | `true` | 是否打印运行日志 |

## 依赖

- `openai>=1.0.0` — LLM API 客户端
- `tiktoken>=0.5.0` — Token 计数
- `numpy>=1.24.0` — 环境依赖
- `scipy>=1.10.0` — 环境依赖
- `pyyaml>=6.0` — YAML 配置支持（可选）
