# Gradient Analysis Documentation: Reward Variance Buckets

This document details the implementation of reward-variance-based gradient analysis in the RAGEN framework.

## Overview
The feature allows for disaggregated gradient analysis across reward variance buckets during PPO training. It enables probing the model's response to specific trajectory types without updating weights. It now supports **component-level gradient norms** (task, entropy, KL) computed per bucket.

## Implementation Details

### 1. Bucket Selection
In `RewardRolloutFilter.split_into_buckets`:
- **Buckets**: `[0-0.2], [0.2-0.5], [0.5-1], [1-2], [2-3], [3-5], [5+]`
- Uses the `reward_std` computed during rollout filtering.

### 2. Analysis Workflow
Integrated in `AgentTrainer.fit`:
1.  **Stable Baselines**: Advantages (GAE) are computed once for the full batch before any probing.
2.  **The Probing Loop**: The trainer iterates through each bucket and calls `actor_rollout_wg.update_actor`.
3.  **No-Update Flag**: Passes `skip_optimizer_step=True` to the actors.
4.  **Component Breakdown**: The actor performs three backward passes (task, entropy, KL) to compute per-component gradient norms.

### 3. Non-Destructive Actor Updates
In `DataParallelPPOActor._optimizer_step`:
- Computes and logs gradient norms for the specific bucket and component.
- Executes `optimizer.zero_grad()` and skips `optimizer.step()` to preserve the starting model state for the next bucket.
  
In `DataParallelPPOActor._update_policy_grad_components` (local actor implementation):
- Performs three backward passes per mini-batch (task / entropy / KL).
- Logs component norms under `actor/grad_norm/{task|entropy|kl}`.

### 4. FSDP & Environment Compatibility
In `fsdp_workers.py`:
- Uses the local actor implementation (`ragen.workers.actor.dp_actor.DataParallelPPOActor`) to enable component-wise gradient analysis **without modifying the verl submodule**.
- **TRL Bypass**: Implemented manual model loading and head attachment to avoid version conflicts with Qwen models.
- **Attention Backend**: Hardcoded `eager` implementation to resolve missing `flash_attn` dependencies in the current environment.

## Usage
Run the analysis using the following flag:
```bash
python3 train.py ... +trainer.gradient_analysis_mode=True
```
Metrics will be logged to WandB and the console under `grad_norm/var_{range}/`.

Component metrics are logged per bucket:
- `grad_norm/<bucket>/task`
- `grad_norm/<bucket>/entropy`
- `grad_norm/<bucket>/kl`
- `grad_norm/<bucket>/loss/{policy|entropy|kl|total}`

---
**Date**: 2026-01-26
**Implementation Status**: Merged and Verified.
