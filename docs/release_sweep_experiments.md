# 四个 Sweep 实验配置总览

本文汇总并对齐以下脚本（以脚本当前实现为准），并对应 `docs/` 中的说明文档：

- `scripts/runs/run_top_p_sweep.sh` ↔ `docs/top_p_sweep.md`
- `scripts/runs/run_entropy_sweep.sh` ↔ `docs/entropy_sweep.md`
- `scripts/runs/run_kl_sweep.sh` ↔ `docs/kl_sweep.md`
- `scripts/runs/run_frozen_lake_slipper_rate_sweep.sh` ↔ `docs/frozen_lake_slipper_sweep.md`

## 0. 共同执行框架

四个脚本都使用并行队列调度（按 `--gpus-per-exp` 分组），并运行：

```bash
python train.py --config-name <CONFIG_NAME> ...
```

共同特征：

- 默认训练步数：`trainer.total_training_steps=400`（可用 `--steps` 覆盖）
- 默认不存 ckpt：`trainer.save_freq=-1`（可用 `--save-freq` 覆盖）
- 优势估计：`algorithm.adv_estimator=gae`
- logger：`trainer.logger=['console','wandb']`
- 训练前验证：`trainer.val_before_train=True`
- 资源参数：`--gpu-memory-utilization`、`--ray-num-cpus`、`--gpus`、`--gpus-per-exp`

## 1. Top-p Sweep（Sokoban）

脚本：`scripts/runs/run_top_p_sweep.sh`  
对应文档：`docs/top_p_sweep.md`

### 实验目的

扫描 `actor_rollout_ref.rollout.rollout_filter_value`（top-p 阈值）对训练效果的影响。

### 关键配置（脚本显式覆盖）

- `--config-name _2_sokoban`
- `model_path=Qwen/Qwen2.5-3B`
- `trainer.project_name=ragen_release_top_p_sweep`
- Sweep 变量：`ROLL_FILTER_VALUES=1.0,0.98,0.95,0.9,0.8,0.6,0.4,nofilter`
- `nofilter` 逻辑：`rollout_filter_value=1.0` 且 `rollout_filter_include_zero=True`
- KL/Entropy：
- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`
- `actor_rollout_ref.actor.entropy_coeff=0.001`
- Filter 相关：
- `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
- `actor_rollout_ref.rollout.rollout_filter_type=largest`
- `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
- `actor_rollout_ref.actor.filter_loss_scaling=none`
- Batch 相关：
- `actor_rollout_ref.actor.ppo_mini_batch_size=32`
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
- `critic.ppo_mini_batch_size=32`
- `critic.ppo_micro_batch_size_per_gpu=4`
- `ppo_mini_batch_size=32`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`
- Env 相关：
- `es_manager.train.env_groups=8`
- `es_manager.train.group_size=16`
- `es_manager.train.env_configs.n_groups=[8]`
- `es_manager.val.env_groups=512`
- `es_manager.val.group_size=1`
- `es_manager.val.env_configs.n_groups=[512]`
- 资源默认值：
- `--gpu-memory-utilization 0.5`
- `--ray-num-cpus 16`

### 输出路径

- 汇总日志：`logs/top_p_sweep_Qwen2.5-3B.log`
- 单实验日志：`logs/top_p_sweep_Qwen2.5-3B/<value_label>/...`
- ckpt：`model_saving/top_p_sweep_Qwen2.5-3B/<value_label>/...`

## 2. Entropy Sweep（Sokoban）

脚本：`scripts/runs/run_entropy_sweep.sh`  
对应文档：`docs/entropy_sweep.md`

### 实验目的

扫描 `actor_rollout_ref.actor.entropy_coeff` 对训练效果的影响。

### 关键配置（脚本显式覆盖）

- `--config-name _2_sokoban`
- `model_path=Qwen/Qwen2.5-3B`
- `trainer.project_name=ragen_release_entropy_sweep`
- Sweep 变量：`ENTROPY_VALUES=0,0.001,0.003,0.01,0.03,0.1`
- KL/Entropy：
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.kl_loss_coef=0`
- `actor_rollout_ref.actor.entropy_coeff=<sweep value>`
- Filter 相关：
- `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
- `actor_rollout_ref.rollout.rollout_filter_value=1`
- `actor_rollout_ref.rollout.rollout_filter_include_zero=True`（默认，可通过参数改）
- `actor_rollout_ref.rollout.rollout_filter_type=largest`
- `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
- `actor_rollout_ref.actor.filter_loss_scaling=none`
- Batch 与 Env 覆盖同 Top-p Sweep（`ppo_mini_batch_size=32`，train env_groups=8，val env_groups=512）
- 资源默认值：
- `--gpu-memory-utilization 0.5`
- `--ray-num-cpus 16`

### 输出路径

- 汇总日志：`logs/entropy_sweep_Qwen2.5-3B.log`
- 单实验日志：`logs/entropy_sweep_Qwen2.5-3B/<filter_tag>/<value_label>/...`
- ckpt：`model_saving/entropy_sweep_Qwen2.5-3B/<value_label>/...`

说明：`filter_tag` 在脚本里为 `nofilter`（`include_zero=True`）或 `filter_zero`（`include_zero=False`）。

## 3. KL Sweep（Sokoban）

脚本：`scripts/runs/run_kl_sweep.sh`  
对应文档：`docs/kl_sweep.md`

### 实验目的

扫描 `actor_rollout_ref.actor.kl_loss_coef` 对训练效果的影响。

### 关键配置（脚本显式覆盖）

- `--config-name _2_sokoban`
- `model_path=Qwen/Qwen2.5-3B`
- `trainer.project_name=ragen_release_kl_sweep`
- Sweep 变量：`KL_VALUES=0,0.001,0.003,0.01,0.03,0.1`
- KL/Entropy：
- `actor_rollout_ref.actor.kl_loss_coef=<sweep value>`
- `actor_rollout_ref.actor.entropy_coeff=0.0`
- `actor_rollout_ref.actor.use_kl_loss=False` 当值为 `0`，否则 `True`
- Filter 相关：
- `actor_rollout_ref.rollout.rollout_filter_strategy=top_p`
- `actor_rollout_ref.rollout.rollout_filter_value=1`
- `actor_rollout_ref.rollout.rollout_filter_include_zero=True`（默认，可通过参数改）
- `actor_rollout_ref.rollout.rollout_filter_type=largest`
- `actor_rollout_ref.rollout.rollout_filter_metric=reward_variance`
- `actor_rollout_ref.actor.filter_loss_scaling=none`
- Batch 与 Env 覆盖同 Top-p Sweep（`ppo_mini_batch_size=32`，train env_groups=8，val env_groups=512）
- 资源默认值：
- `--gpu-memory-utilization 0.5`
- `--ray-num-cpus 16`

### 输出路径

- 汇总日志：`logs/kl_sweep_Qwen2.5-3B.log`
- 单实验日志：`logs/kl_sweep_Qwen2.5-3B/<filter_tag>/<value_label>/...`
- ckpt：`model_saving/kl_sweep_Qwen2.5-3B/<value_label>/...`

## 4. FrozenLake Slipper Rate Sweep

脚本：`scripts/runs/run_frozen_lake_slipper_rate_sweep.sh`  
对应文档：`docs/frozen_lake_slipper_sweep.md`

### 实验目的

在 FrozenLake 上扫描随机性（`slipper_rate`），并比较 `filter` 与 `nofilter` 两种模式。

### 关键配置（脚本显式覆盖）

- `--config-name _3_frozen_lake`
- `model_path=Qwen/Qwen2.5-3B`
- `trainer.project_name=ragen_release_frozenlake_slipper_rate_sweep`
- Sweep 变量：
- `SLIPPER_RATES=0,50,80,90,95,100`
- `FILTER_MODES=filter,nofilter`
- rate 归一化：支持 `50` / `0.5` / `50%` 三种写法
- 环境随机性映射：`success_rate = 1 - slipper_rate`
- Filter 模式参数：
- `filter`: `top_p=0.9`, `rollout_filter_include_zero=False`
- `nofilter`: `top_p=1.0`, `rollout_filter_include_zero=True`
- KL/Entropy：
- `actor_rollout_ref.actor.use_kl_loss=False`
- `actor_rollout_ref.actor.kl_loss_type=low-var-kl`
- `actor_rollout_ref.actor.kl_loss_coef=0`
- `actor_rollout_ref.actor.entropy_coeff=0`
- 其他关键项：
- `actor_rollout_ref.actor.loss_agg_mode=token-mean`
- `actor_rollout_ref.actor.filter_loss_scaling=none`
- `custom_envs.CoordFrozenLake.env_config.success_rate=<computed>`
- `actor_rollout_ref.actor.checkpoint.save_contents=[model]`
- `critic.checkpoint.save_contents=[model]`
- Batch 相关（同上）：`ppo_mini_batch_size=32`、micro batch `4/8`
- 资源默认值：
- `--gpu-memory-utilization 0.5`
- `--ray-num-cpus 16`
- `--cooldown 30`

### 输出路径

- 汇总日志：`logs/frozenlake_slipper_rate_sweep_Qwen2.5-3B.log`
- 单实验日志：`logs/frozenlake_slipper_rate_sweep_Qwen2.5-3B/<mode>/slip<label>/...`
- ckpt：`model_saving/frozenlake_slipper_rate_sweep_Qwen2.5-3B/<mode>/slip<label>/...`

## 5. 配置继承说明（避免误解）

- `_2_sokoban.yaml` 只改了 `trainer.experiment_name`，大部分默认来自 `config/base.yaml`。
- `_3_frozen_lake.yaml` 在 `base` 上把 env tag 切到 `CoordFrozenLake`，并给出默认 `success_rate=1`；slipper sweep 脚本会在运行时覆盖该值。
- 三个 Sokoban sweep 脚本没有显式设置 `algorithm.kl_ctrl.kl_coef`，该值沿用 `config/base.yaml`（当前为 `0.000`）。