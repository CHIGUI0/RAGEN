# Plan

## ✅ 已完成：迁移 rllm Search 环境到 RAGEN

Search 环境已完成迁移，所有测试通过（reward 9/9，env 集成 9/9，retrieval client 2/2）。详见 progress.md。

---

## 🔄 进行中：跑 Search Benchmark 实验

### 术语表

| 术语 | 解释 |
|------|------|
| **Search Benchmark** | 在 SearchQA (HotpotQA + Dense Retrieval) 上对比不同模型/算法的 RL 训练效果 |
| **PPO** | Proximal Policy Optimization，`adv_estimator=gae` + `token-mean` loss |
| **GRPO** | Group Relative Policy Optimization，`adv_estimator=grpo` + `seq-mean-token-mean` loss |
| **filter/nofilter** | 过滤策略：filter = top_p=0.9（丢弃低质量 rollout），nofilter = top_p=1.0（保留所有） |
| **wandb** | Weights & Biases，实验日志和可视化平台 |
| **retrieval server** | 检索服务器，提供 Dense Retrieval 的文档召回，每个 SLURM job 自动在独立端口启动 |

---

### 实验概览

4 组实验，每组对比 filter vs nofilter，共 **8 个独立 SLURM job**：

| # | 模型 | 算法 | filter | nofilter | GPU 数 |
|---|------|------|--------|----------|--------|
| 1 | Qwen2.5-3B-Instruct | PPO | ✅ | ✅ | 1 |
| 2 | Qwen2.5-3B-Instruct | GRPO | ✅ | ✅ | 1 |
| 3 | Qwen2.5-7B-Instruct | PPO | ✅ | ✅ | 2 (TP=2) |
| 4 | Llama-3.2-3B-Instruct | PPO | ✅ | ✅ | 1 |

- 训练步数：400 步
- 分区：`gpuH200x8`（48h 时限）
- wandb project: `ragen_search_benchmark`

---

### 提交方式

**一键提交全部 8 个实验：**
```bash
cd /u/lliu22/RAGEN
bash scripts/slurm/submit_search_benchmark.sh
```

**单独提交某一个实验（示例）：**
```bash
# 组 1: Qwen2.5-3B-Instruct + PPO
sbatch --job-name=search-PPO-filter-Qwen2.5-3B-Instruct \
    --export=ALL,MODEL=Qwen2.5-3B-Instruct,ALGO=PPO,FILTER=filter \
    scripts/slurm/run_search_benchmark.sbatch

sbatch --job-name=search-PPO-nofilter-Qwen2.5-3B-Instruct \
    --export=ALL,MODEL=Qwen2.5-3B-Instruct,ALGO=PPO,FILTER=nofilter \
    scripts/slurm/run_search_benchmark.sbatch

# 组 2: Qwen2.5-3B-Instruct + GRPO
sbatch --job-name=search-GRPO-filter-Qwen2.5-3B-Instruct \
    --export=ALL,MODEL=Qwen2.5-3B-Instruct,ALGO=GRPO,FILTER=filter \
    scripts/slurm/run_search_benchmark.sbatch

sbatch --job-name=search-GRPO-nofilter-Qwen2.5-3B-Instruct \
    --export=ALL,MODEL=Qwen2.5-3B-Instruct,ALGO=GRPO,FILTER=nofilter \
    scripts/slurm/run_search_benchmark.sbatch

# 组 3: Qwen2.5-7B-Instruct + PPO（需要 2 GPU）
sbatch --gpus=2 --job-name=search-PPO-filter-Qwen2.5-7B-Instruct \
    --export=ALL,MODEL=Qwen2.5-7B-Instruct,ALGO=PPO,FILTER=filter \
    scripts/slurm/run_search_benchmark.sbatch

sbatch --gpus=2 --job-name=search-PPO-nofilter-Qwen2.5-7B-Instruct \
    --export=ALL,MODEL=Qwen2.5-7B-Instruct,ALGO=PPO,FILTER=nofilter \
    scripts/slurm/run_search_benchmark.sbatch

# 组 4: Llama-3.2-3B-Instruct + PPO
sbatch --job-name=search-PPO-filter-Llama-3.2-3B-Instruct \
    --export=ALL,MODEL=Llama-3.2-3B-Instruct,ALGO=PPO,FILTER=filter \
    scripts/slurm/run_search_benchmark.sbatch

sbatch --job-name=search-PPO-nofilter-Llama-3.2-3B-Instruct \
    --export=ALL,MODEL=Llama-3.2-3B-Instruct,ALGO=PPO,FILTER=nofilter \
    scripts/slurm/run_search_benchmark.sbatch
```

---

### 每个 SLURM job 的执行流程

由 `run_search_benchmark.sbatch` 统一控制：

1. 激活 conda 环境 `ragen`
2. 下载/准备 search index 数据（首次自动，有文件锁防止重复）
3. 准备 HotpotQA parquet 数据
4. 启动 retrieval server（端口 = 8000 + JOB_ID % 1000，自动避免冲突）
5. 等待 retrieval server 健康检查通过
6. 调用 `run_search_benchmark.sh` 跑训练（400 步）
7. 训练结束后自动清理 retrieval server

### 模型超参差异

| 模型 | GPU 数 | TP | micro_batch | gpu_mem_util |
|------|--------|-----|-------------|--------------|
| 3B 系列 | 1 | 1 | 2 | 0.4 |
| 7B 系列 | 2 | 2 | 1 | 0.4 |

### 输出

- SLURM 日志：`logs/slurm/search_benchmark_<jobid>.out/.err`
- 训练日志：`logs/search_benchmark/search-<algo>-<filter>-<model>.log`
- 结果汇总：`logs/search_benchmark/search-<algo>-<filter>-<model>.result`
- wandb project: `ragen_search_benchmark`

### 涉及的脚本文件

| 脚本 | 作用 |
|------|------|
| `scripts/slurm/submit_search_benchmark.sh` | 一键提交全部 8 个实验 |
| `scripts/slurm/run_search_benchmark.sbatch` | 单个实验的 SLURM 脚本（数据准备 + 启动 retrieval + 训练） |
| `scripts/runs/run_search_benchmark.sh` | 训练执行器（被 sbatch 调用） |
| `scripts/slurm/debug_search.sbatch` | Debug 版（interactive 分区，10 步，1h 时限） |

### 当前状态

- 第一批 8 个实验（3月2日）已全部完成，7/8 失败（OOM 或 prompt 超长），1/8 成功（Llama-3.2-3B PPO filter）
- 需修复后重新提交

---

## 🔄 进行中：修复 Search Benchmark 并重跑

### 问题分析

| 失败原因 | 影响实验 |
|----------|----------|
| OOM（CUDA out of memory） | Qwen2.5-3B PPO×2、Qwen2.5-7B PPO×2 |
| Prompt 超长（5501 > 5500） | Qwen2.5-3B GRPO×2、Llama-3.2-3B PPO nofilter |

Prompt 超长根因：`_apply_max_length` 截断后 `text + "<think>"` 多加 1 token；同时单轮搜索结果本身就可能过长。

### 修改方案（3 处，已完成）

**1. `ragen/env/search/retrieval_client.py`**
- 给 `_format_results` 加 total 字符上限：`max_total_chars=4000`（≈1k tokens）
- 各 doc 截断后，整体输出再截断到 4000 chars，超出部分加 `...`

**2. `ragen/env/search/config.py`**
- 新增字段：`max_total_chars: int = 4000`

**3. `config/_10_search.yaml`**
- `max_model_len: 5500 → 5000`
- `max_num_batched_tokens: 5500 → 5000`

### 术语

| 术语 | 解释 |
|------|------|
| **max_total_chars** | 单次 search 返回的所有文档合并后的字符上限（≈1k tokens），防止单轮 context 过长 |
