# Figure 3：干预 Sweep 实验

三类干预轴——**Entropy 正则**、**KL 约束**、**RV-filter**——在同一 Sokoban 训练设置上独立扫描。第四组实验将 RV-filter 复现到 FrozenLake 环境。

## 可比性协议

每组 sweep 仅改变**一个**干预轴，其余两个固定（或关闭）。三组 Sokoban sweep 共享相同的模型、环境预算、PPO 超参和优势估计器，唯一自由度就是被扫描的旋钮本身。

| Sweep | 扫描变量 | KL | Entropy | RV-filter |
|---|---|---|---|---|
| Entropy | `entropy_coeff` ∈ {0 … 0.1} | 关闭 | **扫描** | 关闭 (value=1, include_zero) |
| KL | `kl_loss_coef` ∈ {0 … 0.1} | **扫描** | 关闭 | 关闭 (value=1, include_zero) |
| RV-filter | `rollout_filter_value` ∈ {1.0 … 0.4} | 0.001（固定） | 0.001（固定） | **扫描** |

每个条件的训练预算：**最多** 400 PPO steps × 128 rollouts/step（8 groups × 16 rollouts/group），受 early stopping 约束（见下节）。

### Early Stopping

训练中存在两个 early stop 条件，任一触发即终止训练（`agent_trainer.py:678-708, 961-977`）：

**条件 A：Reward Variance Collapse。** 前 10 个成功训练 step 的 `rollout/in_group_reward_std` 取均值作为 baseline variance。此后若**连续 10 步**的 variance 均 < baseline 的 10%，则触发。含义：策略已塌缩，所有 rollout 给出几乎相同的 reward。

**条件 B：Validation Success 过低。** 每 `test_freq=10` 步做一次验证。若 `val-env/*/success < 1%` 连续 5 次验证（即跨 50 个训练 step），则触发。含义：模型已崩溃，完全无法完成任务。

**对公平性的影响：** 不同超参条件可能在不同 step 被 early stop，导致实际训练步数不等。但这两个条件检测的都是**训练已经失败**的状态（策略塌缩或完全不学），而非"差不多收敛了就停"。如果某个条件被 early stop，本身就说明该超参配置导致了训练失败——这恰好是 sweep 想要揭示的信息。W&B 日志中 `early_stopped/*=1.0` 标记了哪些 run 被提前终止。

---

## 共享设置（三组 Sokoban Sweep）

| 参数 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-3B`（base，非 Instruct） |
| Config | `_2_sokoban` |
| 优势估计器 | GAE |
| 训练步数 | 400 |
| `ppo_mini_batch_size` | 32（actor + critic） |
| `loss_agg_mode` | `token-mean`（默认） |
| `filter_loss_scaling` | `none` |
| Rollout filter 策略 | `top_p`, type=`largest`, metric=`reward_variance` |
| 训练环境 | `env_groups=8`, `group_size=16`（128 rollouts/step） |
| 验证环境 | `env_groups=512`, `group_size=1` |

---

## 1. Entropy Sweep

**扫描：** `entropy_coeff` ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}。KL 关闭；RV-filter 关闭（`rollout_filter_value=1`, `include_zero=True`）。

### 实现细节

Entropy bonus 是在每个 response token 位置上对**完整词表**计算的 **token 级** Shannon 熵（不限于 reasoning token）：

```
H_t = logsumexp(z_t) − Σ_v softmax(z_t)_v · (z_t)_v
```

其中 `z_t` 是位置 `t` 的 logit 向量（经过 temperature 缩放后）。

- 作用范围：所有 response token（由 `response_mask` 掩码）；prompt token 被排除。
- 聚合方式：`token-mean`——对 batch 内所有未被掩码的 response token 位置取均值。
- 融入 loss：`policy_loss = pg_loss − entropy_coeff × H_mean`，正系数**鼓励**更高的熵（更多探索）。
- 当 `entropy_from_logits_with_chunking=True` 时，计算按 chunk_size=2048 分块以节省显存，数学上完全等价。

来源：`ragen/workers/actor/dp_actor.py:316-341`，`verl/utils/torch_functional.py:145-160`。

---

## 2. KL Sweep

**扫描：** `kl_loss_coef` ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}。Entropy 关闭（`entropy_coeff=0`）；RV-filter 关闭。当 `coef=0` 时，`use_kl_loss=False`（不执行 ref-policy 前向传播）。

### 实现细节

KL 惩罚是**当前策略** π_θ 与**冻结的参考策略** π_ref 之间的 **token 级**散度。π_ref 是训练初始 checkpoint，作为独立 worker 在整个训练过程中保持冻结——**不是**上一轮 PPO 迭代的策略。

默认 `kl_loss_type=kl`（k1 估计器）：

```
KL_t = log π_θ(a_t | s_t) − log π_ref(a_t | s_t)
```

这是 Schulman "Approximating KL Divergence" 博客中的单样本（action 级）KL 估计器，对 E_π_θ[KL] 无偏。

其他可用类型：
- `k2` / `mse`：0.5 × (log π_θ − log π_ref)²，给出 KL **梯度**的无偏估计。
- `low_var_kl` / `k3`：`exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1`，低方差估计器，clamp 到 [−10, 10]。
- 后缀 `+`（如 `k3+`）：straight-through 技巧——前向用指定估计器，反向用 `k2` 以获得无偏梯度。

- 作用范围：所有 response token，与 entropy 相同的 mask。
- 聚合方式：`token-mean`。
- 融入 loss：`policy_loss = pg_loss + kl_loss_coef × KL_mean`（正号惩罚偏离 ref）。

来源：`ragen/workers/actor/dp_actor.py:345-353`，`verl/trainer/ppo/core_algos.py:1412-1471`。

---

## 3. Top-p / RV-Filter Sweep（Sokoban）

**扫描：** `rollout_filter_value` ∈ {1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, nofilter}。KL 和 entropy 都以 0.001 轻度开启（固定）。

### 实现细节

**RV 定义。** 对每个 prompt（env group）`x`，采样 16 条独立 rollout。每组的 reward variance 得分为：

```
s(x) = Std_{i=1..16}[ R(x, y_i) ]
```

其中 `R` 是 `original_rm_scores.sum(dim=-1)`，std 使用 `torch.std(dim=-1)`（Bessel 修正，N−1 分母）。

**过滤策略（`top_p`, `largest`）。** 对 variance 得分做 softmax，降序排列，保留累积概率质量达到阈值 `p` 的最小集合：

1. `probs = softmax(s)`，所有 group 上做 softmax（`largest` → 原始得分作为 logits）。
2. 按 `probs` 降序排列；计算 `cumsum`。
3. 保留累积概率 ≥ `p` 的最小 group 集合。
4. 至少保留一个 group。

因此 `rollout_filter_value=0.9` 保留覆盖了 reward-variance 得分 softmax 质量 90% 的 nucleus groups。

**被过滤 group 的处理。** 被过滤掉的 group **整体丢弃**（`batch.batch = batch.batch[mask]`）。`filter_kept_ratio` 记录在 `meta_info` 中供可选的 loss scaling 使用（但本组 sweep 中 `filter_loss_scaling=none`，不做任何补偿）。

**`include_zero`。** 当 `include_zero=False` 时，|s(x)| < 1e-10 的 group（所有 reward 完全相同）在 top-p 选择**之前**就被排除。`nofilter` 条件使用 `include_zero=True` + `value=1.0`，保留全部。

来源：`ragen/trainer/rollout_filter.py:56-107`（选择），`329-381`（reward variance 计算），`159-180`（masking）。

---

## 4. FrozenLake Top-p / RV-Filter Sweep

**目标：** 测试 RV-filtering 是否能泛化到不同环境（FrozenLake），在两个随机性水平下进行。

| 参数 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-3B-Instruct` |
| Config | `_3_frozen_lake` |
| 扫描变量 | `rollout_filter_value`（top-p 阈值，通过 `--top-p` 指定） |
| `--rand high` | `success_rate=0.8`，project=`ragen_frozenlake_high_rand_top_p_sweep` |
| `--rand low` | `success_rate=0.98`，project=`ragen_frozenlake_low_rand_top_p_sweep` |
| `use_kl_loss` | `False`（KL 关闭） |
| `entropy_coeff` | `0.001` |
| `loss_agg_mode` | `token-mean` |
| `filter_loss_scaling` | `none` |
| `gpu_memory_utilization` | 0.3 |
| 训练步数 | 400 |
| 优势估计器 | GAE |

与 Sokoban top-p sweep 的关键区别：
- KL 关闭（`use_kl_loss=False`）；Sokoban sweep 中 KL 开启。
- 使用 `Instruct` 模型变体。
- 不覆盖 `rollout_filter_type`、`rollout_filter_metric`、`rollout_filter_include_zero`、`ppo_mini_batch_size` 和 env group sizes——均使用 `_3_frozen_lake` config 默认值。
- `gpu_memory_utilization=0.3`（Sokoban 为 0.5）。

---

## 脚本

```
scripts/runs/run_entropy_sweep.sh
scripts/runs/run_kl_sweep.sh
scripts/runs/run_top_p_sweep.sh
scripts/runs/run_frozen_lake_rand_top_p_sweep.sh
```
