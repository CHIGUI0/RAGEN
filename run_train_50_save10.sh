#!/bin/bash
set -euo pipefail

# Train for 50 steps, save every 10 steps (no gradient analysis).

# -----------------------
# GPU AUTO-DETECTION
# -----------------------
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
  else
    echo 0
  fi
}

TOTAL_GPUS=$(detect_gpus)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected via nvidia-smi." >&2
    exit 1
fi
echo "INFO: Detected $TOTAL_GPUS GPUs."

GPU_CSV=$(seq -s, 0 $((TOTAL_GPUS - 1)))

# -----------------------
# Experiment Parameters
# -----------------------
ENV="_2_sokoban"
EXP_NAME="gradient_analysis_ckpt_sokoban_3b"
OUTPUT_DIR="/mnt/permanent/xjin/20260126_filters_final/${EXP_NAME}"
MODEL_PATH="Qwen/Qwen2.5-3B"

mkdir -p "$OUTPUT_DIR"

COMMON_FLAGS=(
  trainer.project_name=AGEN_gradient_analysis
  trainer.experiment_name="${EXP_NAME}"
  trainer.n_gpus_per_node="${TOTAL_GPUS}"
  trainer.nnodes=1
  micro_batch_size_per_gpu=4
  ppo_mini_batch_size=32
  algorithm.kl_ctrl.kl_coef=0.001
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.use_kl_loss=True
  es_manager.train.env_groups=8
  es_manager.train.group_size=16
  es_manager.train.env_configs.n_groups=[8]
  trainer.default_local_dir="${OUTPUT_DIR}"
  model_path="${MODEL_PATH}"
  algorithm.adv_estimator=gae
  system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\""
  trainer.val_before_train=False
)

python3 train.py --config-name "${ENV}" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=50 \
  trainer.save_freq=10 \
  "${COMMON_FLAGS[@]}"
