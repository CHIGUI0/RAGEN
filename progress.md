# Progress Log

---

## 2026-02-27: 迁移 rllm Search 环境到 RAGEN

### 任务描述
将 rllm (`agentica-project/rllm`) 的 Search 环境（HotpotQA 多跳问答 + Dense Retrieval）迁移到 RAGEN 框架。

### 方案决策

**核心设计决策：采用 WebShop 模式**
- 选择 `BaseLanguageBasedEnv` 而非修改 RAGEN 框架支持 tool-calling
- 动作格式：`search[query]` / `finish[answer]`（与 WebShop 的 `search[...]` / `click[...]` 一致）
- 理由：完全兼容现有 ctx_manager，无需改动框架核心

**rllm vs RAGEN 架构差异处理：**

| 差异点 | 处理方式 |
|--------|----------|
| JSON tool_call → 文本 action | 用 `search[...]/finish[...]` 文本格式 |
| 外部 task dict → seed-based | 效仿 CountdownEnv，`seed % len(data)` 选题 |
| httpx → requests | 减少新依赖，用 requests 替代 |
| 5-tuple → 4-tuple | 去掉 truncated，合并为 done |
| RewardInput/Output → 直接返回 | 去掉 rllm 数据类，直接返回 (float, dict) |

**风险应对：**
- 检索服务依赖 → 加入 `mock_mode` + `MockRetrievalClient` + 容错不 crash
- answer 抽取 → 主路径 `finish[...]` 直接解析，保留 `extract_answer_from_response` 作 fallback
- 搜索结果过长 → 每文档截断 300 字符，top_k 可配置

### 代码改动

**新增文件（6 个）：**
```
ragen/env/search/__init__.py          # 模块导出
ragen/env/search/config.py            # SearchEnvConfig dataclass
ragen/env/search/env.py               # SearchEnv 核心（BaseLanguageBasedEnv + gym.Env）
ragen/env/search/reward.py            # F1/EM 奖励函数（从 rllm 迁移）
ragen/env/search/retrieval_client.py  # RetrievalClient + MockRetrievalClient
config/_10_search.yaml                # 训练配置
```

**新增脚本（3 个，从 rllm 复制/改写）：**
```
scripts/retrieval/server.py           # Dense retrieval Flask server（原样复制）
scripts/retrieval/launch_server.sh    # 启动脚本（原样复制）
scripts/prepare_search_data.py        # HotpotQA 数据准备（改写为 RAGEN parquet 格式）
scripts/download_search_index.py      # Wikipedia 索引下载（原样复制）
```

**修改文件（3 个）：**
```
ragen/env/__init__.py                 # 添加 search 环境注册（try/except）
config/envs.yaml                      # 添加 SearchQA + SearchQAMock 条目
setup.py                              # 添加 search extras_require
```

### 从 rllm 迁移的核心逻辑

| 来源 | 目标 | 迁移方式 |
|------|------|----------|
| `rllm/rewards/search_reward.py` → `reward.py` | 保留 F1/EM/normalize/extract 逻辑，去掉 RewardInput/Output/Config 数据类 |
| `examples/search/local_retrieval_tool.py` → `retrieval_client.py` | 去掉 Tool 基类，httpx→requests，加 MockRetrievalClient |
| `rllm/agents/system_prompts.py` SEARCH_SYSTEM_PROMPT → `envs.yaml` env_instruction | `\boxed{}` 格式改为 `<think>/<answer>` + `search[]/finish[]` |

### 测试结果

**reward.py 单元测试（9/9 通过）：**
- exact match、case-insensitive、partial F1、wrong answer、multiple ground truths
- extract_answer_from_response（free text、`\boxed{}` format）
- yes/no 类型、空答案

**SearchEnv 集成测试（9/9 通过，mock 模式）：**
- reset(seed) 确定性选题
- search[query] → 返回搜索结果，done=False，reward=0
- finish[correct] → done=True，reward=1.0
- finish[wrong] → done=True，reward=0.0
- 不同 seed 不同题目
- fallback 自由文本 → 自动抽取答案
- 空 action → invalid
- max_steps 耗尽 → 自动终止
- 相同 seed → 相同 observation

**retrieval_client.py 测试（2/2 通过）：**
- MockRetrievalClient 正常返回
- RetrievalClient 服务不可用时优雅降级，不 crash
