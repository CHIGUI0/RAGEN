#!/bin/bash
# Submit all 8 search benchmark experiments as independent slurm jobs.
#
# Usage:
#   bash scripts/slurm/submit_search_benchmark.sh
#   bash scripts/slurm/submit_search_benchmark.sh --dry-run   # print commands without submitting

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

SBATCH_SCRIPT="scripts/slurm/run_search_benchmark.sbatch"
mkdir -p logs/slurm

# 4 model-algo combinations × 2 filters = 8 experiments
EXPERIMENTS=(
    "Qwen2.5-3B-Instruct|PPO|filter"
    "Qwen2.5-3B-Instruct|PPO|nofilter"
    "Qwen2.5-3B-Instruct|GRPO|filter"
    "Qwen2.5-3B-Instruct|GRPO|nofilter"
    "Qwen2.5-7B-Instruct|PPO|filter"
    "Qwen2.5-7B-Instruct|PPO|nofilter"
    "Llama-3.2-3B-Instruct|PPO|filter"
    "Llama-3.2-3B-Instruct|PPO|nofilter"
)

echo "=== Submitting ${#EXPERIMENTS[@]} search benchmark jobs ==="
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r model algo filter <<< "$exp"
    job_name="search-${algo}-${filter}-${model}"

    # 7B models need 2 GPUs for tensor parallelism
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
