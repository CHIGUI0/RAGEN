# Figure 3: Intervention Sweep Experiments

Three intervention axes — **Entropy regularization**, **KL constraint**, and **RV-filter** — are swept independently on a shared Sokoban training setup. A fourth sweep replicates the RV-filter experiment on FrozenLake.

## Comparability Protocol

Each sweep isolates **exactly one** intervention while holding the other two fixed (or off). All three Sokoban sweeps share the same model, environment budget, PPO hyper-parameters, and advantage estimator, so the only degree of freedom is the swept knob itself.

| Sweep | Swept knob | KL | Entropy | RV-filter |
|---|---|---|---|---|
| Entropy | `entropy_coeff` ∈ {0 … 0.1} | OFF | **swept** | OFF (value=1, include_zero) |
| KL | `kl_loss_coef` ∈ {0 … 0.1} | **swept** | OFF | OFF (value=1, include_zero) |
| RV-filter | `rollout_filter_value` ∈ {1.0 … 0.4} | 0.001 (fixed) | 0.001 (fixed) | **swept** |

Training budget per condition: 400 PPO steps × 128 rollouts/step (8 groups × 16 rollouts/group).

---

## Common Settings (All Three Sokoban Sweeps)

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` (base, not Instruct) |
| Config | `_2_sokoban` |
| Advantage estimator | GAE |
| Training steps | 400 |
| `ppo_mini_batch_size` | 32 (actor + critic) |
| `loss_agg_mode` | `token-mean` (default) |
| `filter_loss_scaling` | `none` |
| Rollout filter strategy | `top_p`, type=`largest`, metric=`reward_variance` |
| Train env | `env_groups=8`, `group_size=16` (128 rollouts/step) |
| Val env | `env_groups=512`, `group_size=1` |

---

## 1. Entropy Sweep

**Swept:** `entropy_coeff` ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}. KL is OFF; RV-filter is OFF (`rollout_filter_value=1`, `include_zero=True`).

### Implementation detail

The entropy bonus is a **token-level** Shannon entropy computed over the **full vocabulary** at each response token position (not restricted to reasoning tokens):

```
H_t = logsumexp(z_t) − Σ_v softmax(z_t)_v · (z_t)_v
```

where `z_t` is the logit vector at position `t` (after temperature scaling).

- Scope: all response tokens (masked by `response_mask`); prompt tokens are excluded.
- Aggregation: `token-mean` — mean over all unmasked response token positions across the batch.
- Integration into loss: `policy_loss = pg_loss − entropy_coeff × H_mean`, so a positive coefficient **encourages** higher entropy (more exploration).
- When `entropy_from_logits_with_chunking=True`, the computation is chunked (chunk_size=2048) for memory efficiency but is mathematically identical.

Source: `ragen/workers/actor/dp_actor.py:316-341`, `verl/utils/torch_functional.py:145-160`.

---

## 2. KL Sweep

**Swept:** `kl_loss_coef` ∈ {0, 0.001, 0.003, 0.01, 0.03, 0.1}. Entropy is OFF (`entropy_coeff=0`); RV-filter is OFF. When `coef=0`, `use_kl_loss=False` (no ref-policy forward pass).

### Implementation detail

The KL penalty is a **token-level** divergence between the **current policy** π_θ and a **frozen reference policy** π_ref (the initial checkpoint, kept as a separate worker throughout training — **not** the previous PPO iteration).

Default `kl_loss_type=kl` (k1 estimator):

```
KL_t = log π_θ(a_t | s_t) − log π_ref(a_t | s_t)
```

This is the single-sample (action-level) KL estimator from Schulman's "Approximating KL Divergence" blog. It is unbiased for E_π_θ[KL].

Other available types:
- `k2` / `mse`: 0.5 × (log π_θ − log π_ref)², gives unbiased **gradient** of KL.
- `low_var_kl` / `k3`: `exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1`, lower-variance estimator, clamped to [−10, 10].
- Suffix `+` (e.g. `k3+`): straight-through trick — forward uses the named estimator, backward uses `k2` for unbiased gradients.

- Scope: all response tokens, same mask as entropy.
- Aggregation: `token-mean`.
- Integration: `policy_loss = pg_loss + kl_loss_coef × KL_mean` (positive sign penalizes divergence from ref).

Source: `ragen/workers/actor/dp_actor.py:345-353`, `verl/trainer/ppo/core_algos.py:1412-1471`.

---

## 3. Top-p / RV-Filter Sweep (Sokoban)

**Swept:** `rollout_filter_value` ∈ {1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, nofilter}. KL and entropy are both lightly on at 0.001 (fixed).

### Implementation detail

**RV definition.** For each prompt (env group) `x`, 16 independent rollouts are sampled. The per-group reward variance score is:

```
s(x) = Std_{i=1..16}[ R(x, y_i) ]
```

where `R` is `original_rm_scores.sum(dim=-1)` and the std is computed with `torch.std(dim=-1)` (Bessel-corrected, N−1 denominator).

**Filtering strategy (`top_p`, `largest`).** Groups are ranked by applying softmax over variance scores, sorted descending, and kept until the cumulative probability mass reaches the threshold `p`:

1. `probs = softmax(s)` over all groups (with `largest` → raw scores as logits).
2. Sort `probs` descending; accumulate `cumsum`.
3. Keep the smallest set of groups whose cumulative probability ≥ `p`.
4. At least one group is always kept.

So `rollout_filter_value=0.9` keeps the nucleus of groups covering 90% of the softmax mass of reward-variance scores.

**What happens to filtered groups.** Filtered-out groups are **dropped entirely** from the batch (`batch.batch = batch.batch[mask]`). The `filter_kept_ratio` is recorded in `meta_info` for optional loss scaling (but `filter_loss_scaling=none` in these sweeps, so no rescaling is applied).

**`include_zero`.** When `include_zero=False`, groups with |s(x)| < 1e-10 (all-identical rewards) are excluded **before** the top-p selection. The `nofilter` condition uses `include_zero=True` + `value=1.0`, keeping everything.

Source: `ragen/trainer/rollout_filter.py:56-107` (selection), `329-381` (reward variance computation), `159-180` (masking).

---

## 4. FrozenLake Top-p / RV-Filter Sweep

**Goal:** Test whether RV-filtering generalizes to a different environment (FrozenLake) under two stochasticity levels.

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Config | `_3_frozen_lake` |
| Swept variable | `rollout_filter_value` (top-p threshold, user-provided via `--top-p`) |
| `--rand high` | `success_rate=0.8`, project=`ragen_frozenlake_high_rand_top_p_sweep` |
| `--rand low` | `success_rate=0.98`, project=`ragen_frozenlake_low_rand_top_p_sweep` |
| `use_kl_loss` | `False` (KL is off) |
| `entropy_coeff` | `0.001` |
| `loss_agg_mode` | `token-mean` |
| `filter_loss_scaling` | `none` |
| `gpu_memory_utilization` | 0.3 |
| Training steps | 400 |
| Advantage estimator | GAE |

Key differences from the Sokoban top-p sweep:
- KL is off (`use_kl_loss=False`); Sokoban sweep has KL on.
- Uses the `Instruct` model variant.
- Does not override `rollout_filter_type`, `rollout_filter_metric`, `rollout_filter_include_zero`, `ppo_mini_batch_size`, or env group sizes — all use `_3_frozen_lake` config defaults.
- `gpu_memory_utilization=0.3` (Sokoban uses 0.5).

---

## Scripts

```
scripts/runs/run_entropy_sweep.sh
scripts/runs/run_kl_sweep.sh
scripts/runs/run_top_p_sweep.sh
scripts/runs/run_frozen_lake_rand_top_p_sweep.sh
```
