# Experiment: Gradient Analysis on Sokoban (1-step)

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

# Single run using all GPUs
python3 train.py --config-name "_2_sokoban" \
    trainer.project_name='AGEN_gradient_analysis' \
    trainer.experiment_name='gradient_analysis_sokoban_3b' \
    trainer.n_gpus_per_node="${TOTAL_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.total_training_steps=1 \
    micro_batch_size_per_gpu=4 \
    ppo_mini_batch_size=32 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.use_kl_loss=True \
    es_manager.train.env_groups=256 \
    es_manager.train.group_size=32 \
    es_manager.train.env_configs.n_groups=[256] \
    trainer.default_local_dir="/mnt/permanent/xjin/20260126_filters_final/gradient_analysis_sokoban_3b" \
    model_path=Qwen/Qwen2.5-3B-Instruct \
    algorithm.adv_estimator=gae \
    system.CUDA_VISIBLE_DEVICES="\"${GPU_CSV}\"" \
    trainer.val_before_train=False \
    +trainer.gradient_analysis_mode=True
