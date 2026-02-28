# KL Sweep Script

`scripts/runs/run_kl_sweep.sh` scans `actor_rollout_ref.actor.kl_loss_coef` on Sokoban while fixing entropy to `0`.

## Script Truth (Current Defaults)

- Config: `_2_sokoban`
- Model: `Qwen2.5-3B`
- Model path: `Qwen/Qwen2.5-3B`
- Project name: `ragen_release_kl_sweep`
- Training steps: `trainer.total_training_steps=400`
- Save frequency: `trainer.save_freq=-1`
- Sweep values: `0,0.001,0.003,0.01,0.03,0.1`
- `rollout_filter_include_zero=True` by default (CLI overridable)
- Algorithm:
- `algorithm.adv_estimator=gae`
- `actor_rollout_ref.actor.kl_loss_coef=<sweep value>`
- `actor_rollout_ref.actor.entropy_coeff=0.0`
- `actor_rollout_ref.actor.use_kl_loss=False` when value is `0`, otherwise `True`
- Filtering:
- `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
- `actor_rollout_ref.rollout.rollout_filter_value=1`
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
bash scripts/runs/run_kl_sweep.sh \
  --kl-values 0,0.001,0.01 \
  --rollout_filter_include_zero False \
  --gpus 0,1 \
  --ray-num-cpus 16 \
  --save-freq 50
```

Options:

- `--kl-values LIST`
- `--rollout_filter_include_zero BOOL`
- `--steps N`
- `--gpus LIST`
- `--gpus-per-exp N`
- `--ray-num-cpus N`
- `--gpu-memory-utilization V`
- `--save-freq N`

## Outputs

- Summary log: `logs/kl_sweep_Qwen2.5-3B.log`
- Per-run logs: `logs/kl_sweep_Qwen2.5-3B/<filter_tag>/<value-label>/`
- Checkpoints: `model_saving/kl_sweep_Qwen2.5-3B/<value-label>/`

`filter_tag` is `nofilter` when `rollout_filter_include_zero=True`, otherwise `filter_zero`.
