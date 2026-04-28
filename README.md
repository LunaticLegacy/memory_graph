# Memory Graph

一个基于 **DAG（有向无环图）** 的 LLM 上下文管理系统。让 Agent 自主决定如何组织对话历史、提取关键记忆、压缩冗长上下文——而不是无脑地把所有历史塞进 prompt。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **DAG 上下文图** | 对话历史以图结构存储，支持多分支、合并、回溯，不再是简单的线性列表。 |
| **Agent 自主决策** | 每轮对话前，LLM Agent 自主决定：加载哪些历史节点、压缩哪些旧上下文、提取哪些关键信息。 |
| **关键记忆（Memory）** | 从对话中提取的重要公式、结论、用户偏好等永久存入独立记忆库，**永远不会被压缩覆盖**，后续可随时检索引用。 |
| **智能压缩** | 自动将久远的对话历史打包成摘要节点，节省 token，同时保留原始节点在图中可追溯。 |
| **多后端 LLM 路由** | 内置 OpenAI / LiteLLM 双后端支持，带自动 fallback 与超时重试。 |

---

## 项目结构

```
memory_graph/
├── modules/
│   ├── llm_fetcher/
│   │   └── llm_fetcher.py      # LLM 多后端调用与流式输出
│   ├── context_graph.py         # DAG 上下文图（节点、边、环检测、祖先链）
│   ├── memory_store.py          # 关键记忆存储（标签检索、文本搜索）
│   ├── context_manager.py       # Agent 决策中枢（选择 / 压缩 / 提取 / 构建上下文）
│   └── __init__.py
├── main.py                      # 可运行的完整 Demo
├── load_sk.sh                   # 本地加载 API Key（已被 .gitignore）
├── .gitignore
└── README.md
```

---

## 快速开始

### 1. 克隆与准备环境

```bash
git clone <repo-url>
cd memory_graph
python -m venv .venv
source .venv/bin/activate
pip install openai  # 或其他 provider，如 litellm
```

### 2. 配置 API Key

```bash
# 新建本地密钥文件（不会被提交到 git）
cat > load_sk.sh << 'EOF'
export DEEPSEEK_API_KEY="sk-your-key-here"
EOF

source ./load_sk.sh
```

> `load_sk.sh` 已默认加入 `.gitignore`，请**不要**将密钥硬编码到代码中。

### 3. 运行 Demo

```bash
python main.py
```

Demo 会演示以下流程：
1. 连续多轮对话建立 DAG 上下文图
2. Agent 自动提取关键公式/结论到 **MemoryStore**
3. Agent 自主选择下轮对话需要加载的历史节点
4. 当历史过长时，Agent 决策压缩旧节点为摘要
5. 最终对话能正确引用之前提取的**记忆**，即使原始上下文已被压缩

---

## 核心概念

### DAG 上下文图

传统对话历史是一条线：`[msg0, msg1, msg2, ...]`。这里是一个**图**：

```
      ┌─→ Node 1 (分支 A: 密码学)
      │
Node 0┤
      │
      └─→ Node 2 (分支 B: 药物研发)
              │
              └─→ Node 3 (合并 A+B 的综合判断)
```

- **Node 0** 是根节点（无父节点）
- **Node 1 / Node 2** 共享同一个父节点，形成对话分支
- **Node 3** 有**两个父节点**，实现分支合并
- 压缩后，旧节点不会删除，而是生成一个**摘要节点**接替链路

### 关键记忆（Memory）

从对话中提取的重要信息（如数学公式、用户偏好、核心结论）会被存入 `MemoryStore`：

```python
memory_store.add_memory(
    content="叠加态公式: |ψ⟩ = α|0⟩ + β|1⟩",
    tags={"公式", "量子计算"}
)
```

这些记忆：
- **独立于上下文图**，即使原始对话被压缩，记忆依然存在
- 每轮对话前，Agent 会主动用关键词检索相关记忆，注入 system prompt

### Agent 决策流程

每轮 `auto_plan=True` 的对话，系统会先发一次"规划请求"给 LLM，输出类似：

```json
{
  "reasoning": "用户问的是测量行为，与 Node 1 的叠加态公式直接相关",
  "selected_node_ids": [1],
  "nodes_to_pack": [0],
  "memories_to_extract": [
    {"content": "测量后坍缩到 |0⟩ 或 |1⟩", "tags": ["测量"]}
  ],
  "memory_queries": ["叠加态", "测量"]
}
```

系统随后自动执行：加载选中节点 → 压缩旧节点 → 提取新记忆 → 检索已有记忆 → 构建最终上下文 → 生成回复。

---

## API 速览

### 最简用法：ContextManager

```python
import asyncio
import os
from modules import LLMFetcher, ContextManager

async def main():
    fetcher = LLMFetcher(
        api_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        model="deepseek-chat",
    )

    # 初始化管理器
    cm = ContextManager(fetcher, max_context_nodes=6)

    # Round 0: 无需决策（图中无节点）
    n0 = await cm.chat("请介绍量子计算的核心思想", auto_plan=False)

    # Round 1: 启用 Agent 自主决策
    n1 = await cm.chat("量子比特的数学表示是什么？", parent_ids={n0}, auto_plan=True)

    # 查看提取的记忆
    for mem in cm.memory_store.get_all_memories():
        print(f"Memory {mem.id}: {mem.content}")

asyncio.run(main())
```

### 直接使用 DAG

```python
from modules import LLMContextGraph, LLMContextPair, LLMContext

graph = LLMContextGraph()

# 添加节点（无父节点 = 根节点）
n0 = graph.add_node(LLMContextPair(
    user_role=LLMContext(role="user", content="你好"),
    llm_role=LLMContext(role="assistant", content="你好！"),
))

# 添加子节点
n1 = graph.add_node(LLMContextPair(...), parent_ids={n0})

# DAG 合并：n2 同时继承 n0 和 n1
n2 = graph.add_node(LLMContextPair(...), parent_ids={n0, n1})

# 获取祖先链
chain = graph.get_ancestor_chain(n2, strategy="longest")  # [0, 1, 2]

# 压平为 LLM 可用的线性上下文
linear = graph.build_linear_context(n2, max_nodes=8)
```

### 手动压缩

```python
summary_id = await graph.compress_ancestors(
    agent=fetcher,
    node_id=n4,
    keep_recent=2,      # 保留最近 2 轮
)
# 返回摘要节点 id，原链前面部分被摘要替代
```

---

## 配置说明

| 环境变量 | 必填 | 说明 |
|----------|------|------|
| `DEEPSEEK_API_KEY` | 是 | LLM API 密钥（或其他 provider 的 key） |

本地开发建议：

```bash
source ./load_sk.sh        # 加载密钥
python main.py             # 运行
```

---

## 扩展方向

- [ ] **向量化记忆检索**：将 MemoryStore 接入向量数据库，实现语义搜索
- [ ] **多 Agent 分支并行**：不同 Agent 在同一 DAG 上探索不同分支，最后合并结论
- [ ] **可视化前端**：用 React/D3 渲染 DAG 结构，交互式回溯对话历史
- [ ] **持久化**：将图和记忆库存入 SQLite/PostgreSQL，支持跨会话恢复

---

## License

MIT
