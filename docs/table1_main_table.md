# Table 1：Main Table 实验

Table 1 的核心目标：证明 RV-aware filtering 不是某个任务/模型/算法的 trick，而是**跨设定都能带来稳定性能增益**的训练干预。

## Filtering 的具体操作

所有实验统一使用同一个 filtering 开关：

| 条件 | `rollout_filter_value` | 含义 |
|---|---|---|
| `filter` | 0.9 | top-p=0.9 的 RV-filter（保留 reward-variance softmax 质量 90% 的 groups） |
| `nofilter` | 1.0 | 保留所有 groups（等价于不过滤） |

---

## 共享设置（三组脚本通用）

| 参数 | 值 |
|---|---|
| 训练步数 | 400（受 early stopping 约束，见 [figure3 文档](figure3_intervention_sweeps.md)） |
| `use_kl_loss` | `False`（KL 关闭） |
| `entropy_coeff` | `0.001` |
| `entropy_from_logits_with_chunking` | `True` |
| `filter_loss_scaling` | `sqrt`（`advantages *= kept_ratio^0.5`） |
| `gpu_memory_utilization` | 0.3 |
| `val_before_train` | `True` |
| FrozenLake 特殊覆盖 | `success_rate=1.0`（确定性环境） |

**KL 完全关闭。** Main table 中 KL 相关的两种机制都是关闭的：

| KL 机制 | 参数 | 状态 | 来源 |
|---|---|---|---|
| **KL Loss**（加到 policy loss 中） | `use_kl_loss=False` | 关闭 | common overrides 显式设置 |
| **KL Reward Penalty**（减到 reward 中） | `use_kl_in_reward=False` | 关闭 | 默认值（`algorithm.py:370`） |

当 `use_kl_in_reward=False` 时，reward 不做任何 KL 扣减，直接 `token_level_rewards = token_level_scores`（`agent_trainer.py:851`）。唯一生效的正则化是 `entropy_coeff=0.001`。

这与 Figure 3 的 Sokoban top-p sweep（`use_kl_loss=True`, `kl_loss_coef=0.001`）不同。

---

## 泛化轴 A：不同算法（`run_main_table_diff_algo.sh`）

**回答质疑：** "RV-filter 是不是只对 PPO 有用？"

- 模型固定：`Qwen/Qwen2.5-3B`
- 任务：Sokoban, FrozenLake, WebShop, MetaMathQA, Countdown
- W&B project：`ragen_main_table_diff_algo`

| 算法 | 优势估计器 | `loss_agg_mode` | 特殊覆盖 |
|---|---|---|---|
| **PPO** | `gae` | `token-mean` | — |
| **DAPO** | `gae` | `token-mean` | `clip_ratio_low=0.2`, `clip_ratio_high=0.28`, 额外显式关闭 KL（`kl_loss_coef=0.0`, `use_kl_in_reward=False`, `kl_ctrl.kl_coef=0.0`） |
| **GRPO** | `grpo` | `seq-mean-token-mean` | `norm_adv_by_std_in_grpo=True` |
| **DrGRPO** | `grpo` | `seq-mean-token-sum` | `norm_adv_by_std_in_grpo=False` |

GRPO 与 DrGRPO 的核心差别在于：GRPO 对 advantage 做组内标准差归一化（`norm_adv_by_std=True`）并用 `seq-mean-token-mean` 聚合 loss；DrGRPO 不做归一化并用 `seq-mean-token-sum`。

---

## 泛化轴 B：不同模型规模（`run_main_table_diff_size.sh`）

**回答质疑：** "是不是只有 3B 才有效？"

- 算法固定：PPO（`gae` + `token-mean`）
- 模型：`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Qwen2.5-7B`
- 任务：Sokoban, FrozenLake, WebShop, MetaMathQA, Countdown
- W&B project：`ragen_main_table_diff_size`

所有规模使用完全相同的超参（包括 `ppo_mini_batch_size` 等由 config 默认值决定的参数）。

---

## 泛化轴 C：不同模型类型（`run_main_table_diff_model.sh`）

**回答质疑：** "是不是 exploit 了 Qwen 的某个特性？"

- 算法固定：PPO（`gae` + `token-mean`）
- 默认模型：`Qwen2.5-3B-Instruct`（可通过 `--models` 传入 `Llama-3.2-3B-Instruct`）
- 任务：Sokoban, FrozenLake, MetaMathQA, Countdown（**无 WebShop**）
- W&B project：`ragen_main_table_diff_model`

模型路径映射：
- `Qwen2.5-3B-Instruct` → `Qwen/Qwen2.5-3B-Instruct`
- `Llama-3.2-3B-Instruct` → `meta-llama/Llama-3.2-3B-Instruct`

---

## 三组脚本的覆盖面总结

| 泛化轴 | 固定 | 变化 | 任务 | 实验数 |
|---|---|---|---|---|
| **A. 算法** | Qwen2.5-3B | PPO / DAPO / GRPO / DrGRPO | 4 任务 | 4 algo × 4 task × 2 filter = 32 |
| **B. 模型规模** | PPO | 0.5B / 1.5B / 3B / 7B | 4 任务 | 4 size × 4 task × 2 filter = 32 |
| **C. 模型类型** | PPO | Instruct / Llama | 4 任务 | 2 model × 4 task × 2 filter = 16|

注意 B 和 C 都只用 PPO，因为算法维度已经被 A 覆盖了。

---

## 任务覆盖

| 任务 | Config | 类型 | 在哪些脚本中 |
|---|---|---|---|
| Sokoban | `_2_sokoban` | 多轮环境交互 | A, B, C |
| FrozenLake | `_3_frozen_lake` | 多轮环境交互（确定性，`success_rate=1.0`） | A, B, C |
| MetaMathQA | `_5_metamathqa` | 推理/解题 | A, B, C |
| Countdown | `_4_countdown` | 推理/解题 | A, B, C |

两类任务风格的覆盖——多轮环境交互（Sokoban, FrozenLake）和推理解题（MetaMathQA, Countdown）——说明 filtering 改善的是"闭环更新的有效学习信号质量"，不依赖任务形式。

---

## 脚本

```
scripts/runs/run_main_table_diff_algo.sh
scripts/runs/run_main_table_diff_size.sh
scripts/runs/run_main_table_diff_model.sh
```
