#!/bin/bash
set -euo pipefail

# Run gradient analysis for 1 step on each saved checkpoint.
# Usage: bash run_grad_analysis_ckpts.sh /path/to/output_dir

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
# Inputs
# -----------------------
OUTPUT_DIR="${1:-/mnt/permanent/xjin/20260126_filters_final/gradient_analysis_ckpt_sokoban_3b}"
ENV="_2_sokoban"
EXP_NAME="gradient_analysis_ckpt_sokoban_3b"
MODEL_PATH="Qwen/Qwen2.5-3B"

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
  es_manager.train.env_groups=256
  es_manager.train.group_size=32
  es_manager.train.env_configs.n_groups=[256]
  trainer.default_local_dir="${OUTPUT_DIR}"
  model_path="${MODEL_PATH}"
  algorithm.adv_estimator=gae
  actor_rollout_ref.rollout.rollout_filter_value=1.0
  system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\""
  trainer.val_before_train=False
)

for step in 10 20 30 40 50; do
  CKPT_DIR="${OUTPUT_DIR}/global_step_${step}"
  if [ ! -d "$CKPT_DIR" ]; then
    echo "WARN: missing checkpoint at $CKPT_DIR, skipping"
    continue
  fi

  python3 train.py --config-name "${ENV}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT_DIR}" \
    +trainer.gradient_analysis_mode=True \
    +trainer.gradient_analysis_every=1 \
    "${COMMON_FLAGS[@]}"

done
