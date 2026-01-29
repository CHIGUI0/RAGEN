#!/bin/bash
set -euo pipefail

# Run GRPO gradient analysis at step 0 using all available GPUs (no resume).
# Usage: bash run_grpo_step0_allgpus.sh

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

GPU_CSV=""
for i in $(seq 0 $((TOTAL_GPUS - 1))); do
  if [ -z "$GPU_CSV" ]; then
    GPU_CSV="${i}"
  else
    GPU_CSV="${GPU_CSV},${i}"
  fi
done

ENV="_2_sokoban"
MODEL_PATH="Qwen/Qwen2.5-3B"
BASE_OUTPUT_DIR="/mnt/permanent/xjin/20260126_filters_final"
ENV_GROUPS=128
GROUP_SIZE=16

EXP_NAME="gradient_analysis_ckpt_sokoban_3b_grpo_${ENV_GROUPS}x${GROUP_SIZE}_step0"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/gradient_analysis_ckpt_sokoban_3b_grpo_step0"

echo "INFO: Launching GRPO step 0 on GPUs [${GPU_CSV}] with exp_name=${EXP_NAME}"

CUDA_VISIBLE_DEVICES="${GPU_CSV}" python3 train.py --config-name "${ENV}" \
  trainer.project_name=AGEN_gradient_analysis \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${TOTAL_GPUS}" \
  trainer.nnodes=1 \
  micro_batch_size_per_gpu=2 \
  ppo_mini_batch_size=32 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_kl_loss=True \
  es_manager.train.env_groups="${ENV_GROUPS}" \
  es_manager.train.group_size="${GROUP_SIZE}" \
  es_manager.train.env_configs.n_groups=["${ENV_GROUPS}"] \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  model_path="${MODEL_PATH}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.rollout.rollout_filter_value=1.0 \
  actor_rollout_ref.rollout.rollout_filter_include_zero=True \
  actor_rollout_ref.rollout.gradient_analysis_bucket_mode=quantile \
  actor_rollout_ref.rollout.gradient_analysis_num_buckets=6 \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.resume_from_path=null \
  +trainer.gradient_analysis_mode=True \
  +trainer.gradient_analysis_every=1 \
  system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\""
