#!/bin/bash
#SBATCH --job-name=debug-search
#SBATCH --partition=gpuH200x8-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/debug_search_%j.out
#SBATCH --error=logs/slurm/debug_search_%j.err
#SBATCH --account=bfea-delta-gpu

# Parameterized debug script for search benchmark
# Submit via: sbatch --export=MODEL=...,ALGO=...,FILTER=... debug_search.sbatch
# Or for 7B: sbatch --gpus=2 --export=MODEL=Qwen2.5-7B-Instruct,ALGO=PPO,FILTER=filter debug_search.sbatch

set -euo pipefail

: "${MODEL:=Qwen2.5-3B-Instruct}"
: "${ALGO:=PPO}"
: "${FILTER:=filter}"

# Per-model settings
case "$MODEL" in
    *7B*)
        GPU_MEM_UTIL=0.4
        NUM_GPUS=2
        TP_SIZE=2
        MICRO_BATCH=1
        ;;
    *)
        GPU_MEM_UTIL=0.4
        NUM_GPUS=1
        TP_SIZE=1
        MICRO_BATCH=2
        ;;
esac

EXP_NAME="search-${ALGO}-${FILTER}-${MODEL}"

RETRIEVAL_PID=""
cleanup() {
    if [ -n "$RETRIEVAL_PID" ] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        kill "$RETRIEVAL_PID" 2>/dev/null || true
        wait "$RETRIEVAL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Debug Search: ${EXP_NAME} ==="
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "Model: $MODEL | Algo: $ALGO | Filter: $FILTER"
echo "GPUs: $NUM_GPUS | TP: $TP_SIZE | micro_batch: $MICRO_BATCH | gpu_mem_util: $GPU_MEM_UTIL"
echo "Date: $(date)"

eval "$(conda shell.bash hook)"
conda activate ragen
cd /u/lliu22/RAGEN

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "=== Verify data ==="
ls -lh ./search_data/prebuilt_indices/corpus.json ./search_data/prebuilt_indices/e5_Flat.index
python scripts/prepare_search_data.py

echo ""
echo "=== Start retrieval server ==="
RETRIEVAL_PORT=$((8000 + SLURM_JOB_ID % 1000))
python scripts/retrieval/server.py --data_dir ./search_data/prebuilt_indices --port $RETRIEVAL_PORT --host 0.0.0.0 &
RETRIEVAL_PID=$!
echo "Server PID=$RETRIEVAL_PID, port=$RETRIEVAL_PORT"

echo "Waiting for health check..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${RETRIEVAL_PORT}/health" 2>/dev/null | grep -q "200"; then
        echo "Server healthy after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "FAIL: Server not healthy after ${MAX_WAIT}s"
    exit 1
fi

echo ""
echo "=== Training: 10 steps ==="
export RETRIEVAL_SERVER_URL="http://127.0.0.1:${RETRIEVAL_PORT}"
# Avoid NFS lock issues: use node-local tmp for datasets cache
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"
bash scripts/runs/run_search_benchmark.sh \
    --steps 10 \
    --models "$MODEL" \
    --algos "$ALGO" \
    --filters "$FILTER" \
    --gpus-per-exp "$NUM_GPUS" \
    --tp "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --micro-batch "$MICRO_BATCH" \
    --retrieval-port "$RETRIEVAL_PORT" \
    --cooldown 0

echo ""
echo "=== Done ${EXP_NAME}: $(date) ==="
