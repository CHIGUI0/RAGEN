#!/bin/bash
set -euo pipefail

# Run gradient analysis for 1 step on each saved checkpoint.
# Usage:
#   bash run_grad_analysis_ckpts.sh /path/to/output_dir [algo] [gpus_per_exp] [steps_csv] [bucket_mode] [num_buckets]
#   bash run_grad_analysis_ckpts.sh {ppo|grpo|drgrpo} [gpus_per_exp] [steps_csv] [bucket_mode] [num_buckets]
#
# Optional env overrides:
#   GRAD_STEPS="20,40,60,80,100"
#   GRAD_BUCKET_MODE="fixed_rv|quantile"
#   GRAD_NUM_BUCKETS="8"

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
  GPUS_PER_EXP="${2:-4}"
  STEPS_CSV="${3:-${GRAD_STEPS:-20,40,60,80,100}}"
  BUCKET_MODE="${4:-${GRAD_BUCKET_MODE:-fixed_rv}}"
  NUM_BUCKETS="${5:-${GRAD_NUM_BUCKETS:-8}}"
else
  OUTPUT_DIR="${1:-${DEFAULT_BASE}/gradient_analysis_ckpt_sokoban_3b_instruct_ppo}"
  ALGO="${2:-ppo}"
  GPUS_PER_EXP="${3:-4}"
  STEPS_CSV="${4:-${GRAD_STEPS:-20,40,60,80,100}}"
  BUCKET_MODE="${5:-${GRAD_BUCKET_MODE:-fixed_rv}}"
  NUM_BUCKETS="${6:-${GRAD_NUM_BUCKETS:-8}}"
fi
if [ "$GPUS_PER_EXP" -le 0 ]; then
  echo "ERROR: gpus_per_exp must be >= 1." >&2
  exit 1
fi
if [ "$GPUS_PER_EXP" -gt "$TOTAL_GPUS" ]; then
  echo "ERROR: gpus_per_exp ($GPUS_PER_EXP) > detected GPUs ($TOTAL_GPUS)." >&2
  exit 1
fi
GPU_GROUPS=()
RUN_GPUS_PER_EXP="$GPUS_PER_EXP"
if [ "$GPUS_PER_EXP" -eq 8 ] && { [ "$ALGO" = "grpo" ] || [ "$ALGO" = "drgrpo" ]; }; then
  RUN_GPUS_PER_EXP=4
  GPU_GROUPS=("0,1,2,3" "4,5,6,7")
else
  GPU_CSV=""
  for i in $(seq 0 $((GPUS_PER_EXP - 1))); do
    if [ -z "$GPU_CSV" ]; then
      GPU_CSV="${i}"
    else
      GPU_CSV="${GPU_CSV},${i}"
    fi
  done
  GPU_GROUPS=("${GPU_CSV}")
fi
ENV="_2_sokoban"
ENV_GROUPS=128
GROUP_SIZE=16
if [ "$ALGO" = "ppo" ]; then
  EXP_NAME_BASE="gradient_analysis_ckpt_sokoban_3b_instruct_ppo_exploratory_${ENV_GROUPS}x${GROUP_SIZE}"
else
  EXP_NAME_BASE="gradient_analysis_ckpt_sokoban_3b_${ALGO}_${ENV_GROUPS}x${GROUP_SIZE}"
fi
MODEL_PATH="Qwen/Qwen2.5-3B"

COMMON_FLAGS=(
  trainer.project_name=AGEN_gradient_analysis
  trainer.n_gpus_per_node="${RUN_GPUS_PER_EXP}"
  trainer.nnodes=1
  micro_batch_size_per_gpu=4
  ppo_mini_batch_size=32
  algorithm.kl_ctrl.kl_coef=0.001
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.use_kl_loss=True
  es_manager.train.env_groups="${ENV_GROUPS}"
  es_manager.train.group_size="${GROUP_SIZE}"
  es_manager.train.env_configs.n_groups=["${ENV_GROUPS}"]
  trainer.default_local_dir="${OUTPUT_DIR}"
  model_path="${MODEL_PATH}"
  actor_rollout_ref.rollout.rollout_filter_value=1.0
  actor_rollout_ref.rollout.rollout_filter_include_zero=True
  actor_rollout_ref.rollout.gradient_analysis_bucket_mode="${BUCKET_MODE}"
  actor_rollout_ref.rollout.gradient_analysis_num_buckets="${NUM_BUCKETS}"
  trainer.val_before_train=False
)

case "$ALGO" in
  ppo)
    COMMON_FLAGS+=(algorithm.adv_estimator=gae actor_rollout_ref.actor.loss_agg_mode=token-mean)
    ;;
  grpo)
    COMMON_FLAGS+=(micro_batch_size_per_gpu=2 algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=true actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean)
    ;;
  drgrpo)
    COMMON_FLAGS+=(micro_batch_size_per_gpu=2 algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=false actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum)
    ;;
  *)
    echo "ERROR: Unknown algo '$ALGO'. Use ppo, grpo, or drgrpo." >&2
    exit 1
    ;;
esac

IFS=',' read -r -a STEPS <<< "${STEPS_CSV}"
if [ "${#GPU_GROUPS[@]}" -gt 1 ]; then
  for ((i=0; i<${#STEPS[@]}; i+=${#GPU_GROUPS[@]})); do
    step_a="${STEPS[$i]:-}"
    step_b="${STEPS[$((i + 1))]:-}"
    if [ -n "${step_a}" ] || [ -n "${step_b}" ]; then
      echo "INFO: Launching step batch: ${step_a:-none} on ${GPU_GROUPS[0]} | ${step_b:-none} on ${GPU_GROUPS[1]}"
    fi
    pids=()
    for ((g=0; g<${#GPU_GROUPS[@]}; g++)); do
      step="${STEPS[$((i + g))]:-}"
      step="${step//[[:space:]]/}"
      if [ -z "${step}" ]; then
        continue
      fi
      CKPT_DIR="${OUTPUT_DIR}/global_step_${step}"
      if [ "$step" != "0" ] && [ ! -d "$CKPT_DIR" ]; then
        echo "WARN: missing checkpoint at $CKPT_DIR, skipping"
        continue
      fi
      EXP_NAME="${EXP_NAME_BASE}_step${step}_bm${BUCKET_MODE}_nb${NUM_BUCKETS}"
      GPU_CSV="${GPU_GROUPS[$g]}"
      EXP_NAME_RUN="${EXP_NAME}_g${GPU_CSV//,/}"

      echo "INFO: Launching step ${step} on GPUs [${GPU_CSV}] with exp_name=${EXP_NAME_RUN}"
      if [ "$step" = "0" ]; then
        echo "INFO: step 0 selected; running from init without resume_from_path"
        STEP0_DIR="${OUTPUT_DIR}_step0"
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 train.py --config-name "${ENV}" \
          trainer.total_epochs=1 \
          trainer.total_training_steps=1 \
          trainer.save_freq=-1 \
          trainer.test_freq=-1 \
          trainer.resume_mode=disable \
          trainer.resume_from_path=null \
          +trainer.gradient_analysis_mode=True \
          +trainer.gradient_analysis_every=1 \
          trainer.experiment_name="${EXP_NAME_RUN}" \
          trainer.default_local_dir="${STEP0_DIR}" \
          system.CUDA_VISIBLE_DEVICES="\"0,1,2,3,4,5,6,7\"" \
          "${COMMON_FLAGS[@]}" \
          trainer.n_gpus_per_node=8 &
      else
        CUDA_VISIBLE_DEVICES="${GPU_CSV}" python3 train.py --config-name "${ENV}" \
          trainer.total_epochs=1 \
          trainer.total_training_steps=1 \
          trainer.save_freq=-1 \
          trainer.test_freq=-1 \
          trainer.resume_mode=resume_path \
          trainer.resume_from_path="${CKPT_DIR}" \
          +trainer.gradient_analysis_mode=True \
          +trainer.gradient_analysis_every=1 \
          trainer.experiment_name="${EXP_NAME_RUN}" \
          system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\"" \
          "${COMMON_FLAGS[@]}" &
      fi
      pids+=($!)
    done
    if [ "${#pids[@]}" -gt 0 ]; then
      wait "${pids[@]}"
    fi
  done
else
  for step in "${STEPS[@]}"; do
    step="${step//[[:space:]]/}"
    CKPT_DIR="${OUTPUT_DIR}/global_step_${step}"
    if [ "$step" != "0" ] && [ ! -d "$CKPT_DIR" ]; then
      echo "WARN: missing checkpoint at $CKPT_DIR, skipping"
      continue
    fi
    EXP_NAME="${EXP_NAME_BASE}_step${step}_bm${BUCKET_MODE}_nb${NUM_BUCKETS}"
    GPU_CSV="${GPU_GROUPS[0]}"
    EXP_NAME_RUN="${EXP_NAME}"

    echo "INFO: Launching step ${step} on GPUs [${GPU_CSV}] with exp_name=${EXP_NAME_RUN}"
    if [ "$step" = "0" ]; then
      echo "INFO: step 0 selected; running from init without resume_from_path"
      STEP0_DIR="${OUTPUT_DIR}_step0"
      CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 train.py --config-name "${ENV}" \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.resume_mode=disable \
        trainer.resume_from_path=null \
        +trainer.gradient_analysis_mode=True \
        +trainer.gradient_analysis_every=1 \
        trainer.experiment_name="${EXP_NAME_RUN}" \
        trainer.default_local_dir="${STEP0_DIR}" \
        system.CUDA_VISIBLE_DEVICES="\"0,1,2,3,4,5,6,7\"" \
        "${COMMON_FLAGS[@]}" \
        trainer.n_gpus_per_node=8
    else
      CUDA_VISIBLE_DEVICES="${GPU_CSV}" python3 train.py --config-name "${ENV}" \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="${CKPT_DIR}" \
        +trainer.gradient_analysis_mode=True \
        +trainer.gradient_analysis_every=1 \
        trainer.experiment_name="${EXP_NAME_RUN}" \
        system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\"" \
        "${COMMON_FLAGS[@]}"
    fi
  done
fi
