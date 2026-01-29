#!/bin/bash
set -euo pipefail

# Run gradient analysis for 1 step on each saved checkpoint.
# Usage: bash run_grad_analysis_ckpts.sh /path/to/output_dir [algo] [gpu_csv] [gpus_per_exp]

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

# -----------------------
# Inputs
# -----------------------
DEFAULT_BASE="/mnt/permanent/xjin/20260126_filters_final"
if [ "${1:-}" = "ppo" ] || [ "${1:-}" = "grpo" ] || [ "${1:-}" = "drgrpo" ]; then
  ALGO="$1"
  if [ "$ALGO" = "ppo" ]; then
    OUTPUT_DIR="${DEFAULT_BASE}/gradient_analysis_ckpt_sokoban_3b_instruct_ppo"
  else
    OUTPUT_DIR="${DEFAULT_BASE}/gradient_analysis_ckpt_sokoban_3b_${ALGO}"
  fi
  GPU_CSV="${2:-0,1,2,3}"
  GPUS_PER_EXP="${3:-4}"
else
  OUTPUT_DIR="${1:-${DEFAULT_BASE}/gradient_analysis_ckpt_sokoban_3b_instruct_ppo}"
  ALGO="${2:-ppo}"
  GPU_CSV="${3:-0,1,2,3}"
  GPUS_PER_EXP="${4:-4}"
fi
ENV="_2_sokoban"
if [ "$ALGO" = "ppo" ]; then
  EXP_NAME_BASE="gradient_analysis_ckpt_sokoban_3b_instruct_ppo_exploratory_32x8"
else
  EXP_NAME_BASE="gradient_analysis_ckpt_sokoban_3b_${ALGO}"
fi
MODEL_PATH="Qwen/Qwen2.5-3B"

COMMON_FLAGS=(
  trainer.project_name=AGEN_gradient_analysis
  trainer.n_gpus_per_node="${GPUS_PER_EXP}"
  trainer.nnodes=1
  micro_batch_size_per_gpu=4
  ppo_mini_batch_size=32
  algorithm.kl_ctrl.kl_coef=0.001
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.use_kl_loss=True
  es_manager.train.env_groups=32
  es_manager.train.group_size=8
  es_manager.train.env_configs.n_groups=[32]
  trainer.default_local_dir="${OUTPUT_DIR}"
  model_path="${MODEL_PATH}"
  algorithm.adv_estimator=grpo
  actor_rollout_ref.rollout.rollout_filter_value=1.0
  actor_rollout_ref.rollout.gradient_analysis_bucket_mode=fixed_rv
  system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\""
  trainer.val_before_train=False
)

case "$ALGO" in
  ppo)
    COMMON_FLAGS+=(algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean)
    ;;
  grpo)
    COMMON_FLAGS+=(algorithm.norm_adv_by_std_in_grpo=true actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean)
    ;;
  drgrpo)
    COMMON_FLAGS+=(algorithm.norm_adv_by_std_in_grpo=false actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum)
    ;;
  *)
    echo "ERROR: Unknown algo '$ALGO'. Use ppo, grpo, or drgrpo." >&2
    exit 1
    ;;
esac

for step in 25 50 75 100; do
  CKPT_DIR="${OUTPUT_DIR}/global_step_${step}"
  if [ ! -d "$CKPT_DIR" ]; then
    echo "WARN: missing checkpoint at $CKPT_DIR, skipping"
    continue
  fi

  EXP_NAME="${EXP_NAME_BASE}_step${step}"

  python3 train.py --config-name "${ENV}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${CKPT_DIR}" \
    +trainer.gradient_analysis_mode=True \
    +trainer.gradient_analysis_every=1 \
    trainer.experiment_name="${EXP_NAME}" \
    "${COMMON_FLAGS[@]}"

done
