# Top-p Sweep Script

`scripts/runs/run_top_p_sweep.sh` scans `actor_rollout_ref.rollout.rollout_filter_value` (top-p threshold) on Sokoban.

## Script Truth (Current Defaults)

- Config: `_2_sokoban`
- Model: `Qwen2.5-3B`
- Model path: `Qwen/Qwen2.5-3B`
- Project name: `ragen_release_top_p_sweep`
- Training steps: `trainer.total_training_steps=400`
- Save frequency: `trainer.save_freq=-1`
- Sweep values: `1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter`
- For `nofilter`: `rollout_filter_value=1.0` and `rollout_filter_include_zero=True`
- Algorithm:
- `algorithm.adv_estimator=gae`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `actor_rollout_ref.actor.entropy_coeff=0.001`
- Filtering:
- `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
- `actor_rollout_ref.rollout.rollout_filter_type=largest`
- `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
- `actor_rollout_ref.actor.filter_loss_scaling=none`
- Batch:
- `actor_rollout_ref.actor.ppo_mini_batch_size=32`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
- `critic.ppo_mini_batch_size=32`
- `critic.ppo_micro_batch_size_per_gpu=4`
- `ppo_mini_batch_size=32`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`
- Environment:
- `es_manager.train.env_groups=8`
- `es_manager.train.group_size=16`
- `es_manager.train.env_configs.n_groups=[8]`
- `es_manager.val.env_groups=512`
- `es_manager.val.group_size=1`
- `es_manager.val.env_configs.n_groups=[512]`
- Resource defaults:
- `--gpu-memory-utilization 0.5`
- `--ray-num-cpus 16`
- `--gpus-per-exp 1`

Note: `algorithm.kl_ctrl.kl_coef` is not overridden in this script and inherits from base config.

## Running

```bash
bash scripts/runs/run_top_p_sweep.sh \
  --rollout_filter_value 0.9,0.8,0.6,nofilter \
  --gpus 0,1 \
  --gpus-per-exp 1 \
  --ray-num-cpus 16 \
  --save-freq 50
```

Options:

- `--steps N`
- `--rollout_filter_value LIST`
- `--gpus LIST`
- `--gpus-per-exp N`
- `--ray-num-cpus N`
- `--gpu-memory-utilization V`
- `--save-freq N`

## Outputs

- Summary log: `logs/top_p_sweep_Qwen2.5-3B.log`
- Per-value logs: `logs/top_p_sweep_Qwen2.5-3B/<value-label>/`
- Checkpoints: `model_saving/top_p_sweep_Qwen2.5-3B/<value-label>/`
