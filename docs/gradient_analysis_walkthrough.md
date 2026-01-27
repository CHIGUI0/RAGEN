# Gradient Analysis Documentation: Reward Variance Buckets

This document details the implementation of reward-variance-based gradient analysis in the RAGEN framework.

## Overview
The feature allows for disaggregated gradient analysis across reward-variance **ranked buckets** during PPO training. It enables probing the model's response to specific trajectory types without updating weights. It now supports **component-level gradient norms** (task, entropy, KL) computed per bucket, and runs as a **periodic reporting module** while normal training proceeds.

## Implementation Details

### 1. Bucket Selection
In `RewardRolloutFilter.split_into_buckets`:
- **Buckets**: configurable via `gradient_analysis_num_buckets` (default: 4) over **groups**, sorted by group-level `reward_std`.
- **Bucket names**: `bucket_1`, `bucket_2`, `bucket_3`, `bucket_4` (when using 4 buckets), ordered low → high reward variance.
- Uses the `reward_std` computed during rollout filtering.
- **Remainder Handling**: if groups don't divide evenly, the remainder is assigned to the **last (highest RV)** bucket so early buckets have equal group counts.

### 2. Training + Reporting Workflow
Integrated in `AgentTrainer.fit`:
1.  **Normal Training**: The actor update always runs on the full batch (no bucketing for the actual weight update).
2.  **Periodic Reporting**: Every `gradient_analysis_every` steps (default: off), the trainer invokes a separate reporter module.
3.  **The Probing Loop**: The reporter iterates through each bucket and calls `actor_rollout_wg.update_actor`.
4.  **No-Update Flag**: Passes `skip_optimizer_step=True` to the actors.
5.  **Component Breakdown**: The actor performs three backward passes (task, entropy, KL) to compute per-component gradient norms.
6.  **Normalized Metrics**: Per-bucket grad norms are also reported per-sample and per-token.

### 3. Non-Destructive Actor Updates
In `DataParallelPPOActor._optimizer_step` (reporting path only):
- Computes and logs gradient norms for the specific bucket and component.
- Executes `optimizer.zero_grad()` and skips `optimizer.step()` to preserve the starting model state for the next bucket.
  
In `DataParallelPPOActor._update_policy_grad_components` (local actor implementation):
- Performs three backward passes per mini-batch (task / entropy / KL).
- Logs component norms under `actor/grad_norm/{task|entropy|kl}`.
  - Component losses are logged in their own pass under `actor/loss/{entropy|kl}`.

### 4. Separate Reporting Module
In `ragen/trainer/gradient_reporter.py`:
- Encapsulates the bucketing loop and metric prefixing.
- Runs only on reporting steps; it never replaces the normal actor update.
- Logs extra metrics:
  - `grad_norm/<bucket>/reward_std_mean`
  - `grad_norm/<bucket>/per_sample/<component>`
  - `grad_norm/<bucket>/per_token/<component>`

### 5. FSDP & Environment Compatibility
In `fsdp_workers.py`:
- Uses the local actor implementation (`ragen.workers.actor.dp_actor.DataParallelPPOActor`) to enable component-wise gradient analysis **without modifying the verl submodule**.
- **TRL Bypass**: Implemented manual model loading and head attachment to avoid version conflicts with Qwen models.
- **Attention Backend**: Hardcoded `eager` implementation to resolve missing `flash_attn` dependencies in the current environment.

## Usage
Run the analysis using the following flags:
```bash
python3 train.py ... +trainer.gradient_analysis_mode=True +trainer.gradient_analysis_every=10
```
Optional flags:
- `+actor_rollout_ref.rollout.gradient_analysis_num_buckets=4`

`gradient_analysis_every` controls the reporting cadence (default: off). Reporting runs on steps where `global_steps % gradient_analysis_every == 0`.
Metrics will be logged to WandB and the console under `grad_norm/<bucket>/`.

Component metrics are logged per bucket:
- `grad_norm/<bucket>/task`
- `grad_norm/<bucket>/entropy`
- `grad_norm/<bucket>/kl`
- `grad_norm/<bucket>/loss/{policy|entropy|kl|total}`
- `grad_norm/<bucket>/per_sample/{task|entropy|kl}`
- `grad_norm/<bucket>/per_token/{task|entropy|kl}`
- `grad_norm/<bucket>/reward_std_mean`

---
**Date**: 2026-01-27
**Implementation Status**: Merged and Verified.
