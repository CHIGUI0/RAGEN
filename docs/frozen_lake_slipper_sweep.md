# FrozenLake Slipper Sweep Script

`scripts/runs/run_frozen_lake_slipper_rate_sweep.sh` runs a FrozenLake stochasticity sweep by varying `slipper_rate`, while comparing `filter` vs `nofilter` settings under the same training budget.

`slipper_rate` is converted to environment `success_rate` as:

- `success_rate = 1 - slipper_rate`

So larger `slipper_rate` means more transition randomness.

## Defaults

- Model: `Qwen2.5-3B`
- Config: `_3_frozen_lake`
- Project name: `ragen_release_frozenlake_slipper_rate_sweep`
- Training steps: `400`
- Slipper sweep: `0,50,80,90,95,100`
- Modes: `filter,nofilter`
- Filter setup:
  - `filter`: `top_p=0.9`, `rollout_filter_include_zero=False`
  - `nofilter`: `top_p=1.0`, `rollout_filter_include_zero=True`
- KL/Entropy:
  - `actor_rollout_ref.actor.use_kl_loss=False`
  - `actor_rollout_ref.actor.kl_loss_coef=0`
  - `actor_rollout_ref.actor.entropy_coeff=0`
- Batch settings:
  - `actor.ppo_micro_batch_size_per_gpu=4`
  - `ref.log_prob_micro_batch_size_per_gpu=8`
  - `rollout.log_prob_micro_batch_size_per_gpu=8`
- Resource defaults:
  - `--gpu-memory-utilization 0.5`
  - `--ray-num-cpus 16`
  - `--save-freq -1`

## Naming Convention

Current naming only records `slip` (not `sr`) and uses compact decimal labels:

- `0.500000` -> `slip0p5`
- `0.950000` -> `slip0p95`

Experiment/log name format:

- `frozenlake_<mode>_slip<label>-<MODEL_NAME>`

Examples:

- `frozenlake_nofilter_slip0p5-Qwen2.5-3B`
- `frozenlake_filter_slip0p95-Qwen2.5-3B`

## Running

Run all default settings:

```bash
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh
```

Run one mode and a custom subset:

```bash
bash scripts/runs/run_frozen_lake_slipper_rate_sweep.sh \
  --slipper-rate 50,80,95 \
  --filter-modes nofilter \
  --gpus 0 \
  --ray-num-cpus 8
```

`--slipper-rate` accepts:

- percentages: `50,80,95`
- ratios: `0.5,0.8,0.95`
- percent suffix: `50%,80%,95%`

## Outputs

- Per-run logs/results:
  - `logs/frozenlake_slipper_rate_sweep_<MODEL_NAME>/<mode>/slip<label>/`
- Sweep summary log:
  - `logs/frozenlake_slipper_rate_sweep_<MODEL_NAME>.log`
- Checkpoints:
  - `model_saving/frozenlake_slipper_rate_sweep_<MODEL_NAME>/<mode>/slip<label>/`
