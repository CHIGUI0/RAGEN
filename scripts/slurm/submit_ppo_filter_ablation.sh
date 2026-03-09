#!/bin/bash
# Submit 3 PPO filter ablation experiments as independent slurm jobs.
#
# Experiments:
#   1. PPO + TopP Filter (top_p=0.9)
#   2. PPO + No Filter (top_p=1.0)
#   3. PPO + TopK Filter (top_k=0.25)
#
# Usage:
#   bash scripts/slurm/submit_ppo_filter_ablation.sh
#   bash scripts/slurm/submit_ppo_filter_ablation.sh --dry-run
#   bash scripts/slurm/submit_ppo_filter_ablation.sh --model Llama-3.2-3B-Instruct

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=false
MODEL="Qwen2.5-3B-Instruct"

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SBATCH_SCRIPT="scripts/slurm/run_search_benchmark.sbatch"
mkdir -p logs/slurm

EXPERIMENTS=(
    "${MODEL}|PPO|filter"
    "${MODEL}|PPO|nofilter"
    "${MODEL}|PPO|topk"
)

echo "=== Submitting ${#EXPERIMENTS[@]} PPO filter ablation jobs (model: ${MODEL}) ==="
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model algo filter <<< "$exp"
    job_name="search-${algo}-${filter}-${model}"

    case "$model" in
        *7B*) gpu_override="--gpus=2" ;;
        *)    gpu_override="" ;;
    esac

    cmd="sbatch ${gpu_override} --job-name=${job_name} --export=ALL,MODEL=${model},ALGO=${algo},FILTER=${filter} ${SBATCH_SCRIPT}"

    if [ "$DRY_RUN" = true ]; then
        echo "[dry-run] $cmd"
    else
        echo -n "Submitting ${job_name} ... "
        $cmd
    fi
done

echo ""
echo "=== All jobs submitted. Use 'squeue -u \$USER' to check status ==="
