#!/bin/bash
set -euo pipefail

# Algorithm parameters
GRPO_MI_NO_FILTER="algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
GRPO_MI_FILTER="algorithm.adv_estimator=grpo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"

PPO_MI_NO_FILTER="algorithm.adv_estimator=ppo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=1"
PPO_MI_FILTER="algorithm.adv_estimator=ppo collapse_detection.enabled=true actor_rollout_ref.rollout.rollout_filter_ratio=0.5"

LOG_DIR="logs/sokoban_mi_runs"
mkdir -p "${LOG_DIR}"

# Sokoban 3B GRPO Mutual Info Runs
python train.py --config-name "_2_sokoban" trainer.experiment_name="sokoban_mi_grpo_no_filter" ${GRPO_MI_NO_FILTER} 2>&1 | tee "${LOG_DIR}/sokoban_mi_grpo_no_filter_thinking.log"

python train.py --config-name "_2_sokoban" trainer.experiment_name="sokoban_mi_grpo_filter0.5" ${GRPO_MI_FILTER} 2>&1 | tee "${LOG_DIR}/sokoban_mi_grpo_filter0.5_thinking.log"

# Sokoban 3B PPO Mutual Info Runs
python train.py --config-name "_2_sokoban" trainer.experiment_name="sokoban_mi_ppo_no_filter" ${PPO_MI_NO_FILTER} 2>&1 | tee "${LOG_DIR}/sokoban_mi_ppo_no_filter_thinking.log"

python train.py --config-name "_2_sokoban" trainer.experiment_name="sokoban_mi_ppo_filter0.5" ${PPO_MI_FILTER} 2>&1 | tee "${LOG_DIR}/sokoban_mi_ppo_filter0.5_thinking.log"