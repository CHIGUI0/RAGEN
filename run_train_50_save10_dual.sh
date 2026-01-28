#!/bin/bash
set -euo pipefail

# Run two algorithms in parallel (default: grpo + drgrpo), splitting GPUs.
# Usage: bash run_train_50_save10_dual.sh [gpus_per_exp]

GPUS_PER_EXP="${1:-4}"

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
if [ "$TOTAL_GPUS" -lt $((GPUS_PER_EXP * 2)) ]; then
    echo "ERROR: Need at least $((GPUS_PER_EXP * 2)) GPUs for two experiments." >&2
    exit 1
fi

# -----------------------
# GPU POOL
# -----------------------
GPU_POOL_FIFO="/tmp/gpu_pool_train_50_$$"
mkfifo "$GPU_POOL_FIFO"
exec 3<>"$GPU_POOL_FIFO"
rm "$GPU_POOL_FIFO"

for ((i=0; i<TOTAL_GPUS; i++)); do
    echo "$i" >&3
done

# -----------------------
# Experiment Parameters
# -----------------------
ENV="_2_sokoban"
MODEL_PATH="Qwen/Qwen2.5-3B"
BASE_OUTPUT_DIR="/mnt/permanent/xjin/20260126_filters_final"
ALGORITHMS=("grpo" "drgrpo")

run_one() {
  local algo="$1"
  local gpu_csv="$2"
  local exp_name="gradient_analysis_ckpt_sokoban_3b_${algo}"
  local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"

  mkdir -p "$output_dir"

  case "$algo" in
    grpo)
      ADV_ESTIMATOR="grpo"
      NORM_ADV_BY_STD_IN_GRPO="true"
      LOSS_AGG_MODE="seq-mean-token-mean"
      ;;
    drgrpo)
      ADV_ESTIMATOR="grpo"
      NORM_ADV_BY_STD_IN_GRPO="false"
      LOSS_AGG_MODE="seq-mean-token-sum"
      ;;
    *)
      echo "ERROR: Unknown algo '$algo'" >&2
      return 1
      ;;
  esac

  python3 train.py --config-name "${ENV}" \
    trainer.project_name=AGEN_gradient_analysis \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPUS_PER_EXP}" \
    trainer.nnodes=1 \
    micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=32 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    es_manager.train.env_groups=8 \
    es_manager.train.group_size=16 \
    es_manager.train.env_configs.n_groups=[8] \
    trainer.default_local_dir="${output_dir}" \
    model_path="${MODEL_PATH}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.norm_adv_by_std_in_grpo="${NORM_ADV_BY_STD_IN_GRPO}" \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    system.CUDA_VISIBLE_DEVICES="\"${gpu_csv}\"" \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    trainer.save_freq=10
}

for algo in "${ALGORITHMS[@]}"; do
  allocated_gpus=()
  for ((i=0; i<GPUS_PER_EXP; i++)); do
    read -u 3 gid
    allocated_gpus+=("$gid")
  done
  gpu_csv=$(IFS=,; echo "${allocated_gpus[*]}")

  (
    echo "Running ${algo} on GPUs ${gpu_csv}"
    if ! run_one "$algo" "$gpu_csv"; then
      echo "ERROR: ${algo} failed on GPUs ${gpu_csv}" >&2
    fi
    for gid in "${allocated_gpus[@]}"; do
      echo "$gid" >&3
    done
  ) &
done

wait
exec 3>&-
