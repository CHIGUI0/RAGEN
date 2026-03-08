#!/bin/bash
#SBATCH --job-name=ragen-search
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=15:00:00
#SBATCH --output=logs/slurm/search_benchmark_%j.out
#SBATCH --error=logs/slurm/search_benchmark_%j.err
#SBATCH --account=bfea-delta-gpu

# === Single Search Benchmark Experiment ===
# Parameterized via environment variables:
#   MODEL  - model name (e.g. Qwen2.5-3B-Instruct)
#   ALGO   - algorithm (PPO or GRPO)
#   FILTER - filter mode (filter or nofilter)
#
# Submit via: sbatch --export=MODEL=...,ALGO=...,FILTER=... run_search_benchmark.sbatch
# Or use: bash scripts/slurm/submit_search_benchmark.sh

set -euo pipefail

# Validate required env vars
: "${MODEL:?ERROR: MODEL env var is required (e.g. Qwen2.5-3B-Instruct)}"
: "${ALGO:?ERROR: ALGO env var is required (PPO or GRPO)}"
: "${FILTER:?ERROR: FILTER env var is required (filter or nofilter)}"

EXP_NAME="search-${ALGO}-${FILTER}-${MODEL}"

# GPU/TP/micro_batch settings per model size
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

# Use unique port per job to avoid conflicts on shared nodes
RETRIEVAL_PORT=$((8000 + SLURM_JOB_ID % 1000))
RETRIEVAL_PID=""

cleanup() {
    echo "=== Cleanup ==="
    if [ -n "$RETRIEVAL_PID" ] && kill -0 "$RETRIEVAL_PID" 2>/dev/null; then
        echo "Stopping retrieval server (PID=$RETRIEVAL_PID)..."
        kill "$RETRIEVAL_PID" 2>/dev/null || true
        wait "$RETRIEVAL_PID" 2>/dev/null || true
        echo "Retrieval server stopped."
    fi
}
trap cleanup EXIT

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Experiment: ${EXP_NAME}"
echo "Retrieval port: ${RETRIEVAL_PORT}"

eval "$(conda shell.bash hook)"
conda activate ragen
cd /u/lliu22/RAGEN

echo "=== GPU Check ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

mkdir -p logs/slurm logs/search_benchmark

# === Step 1: Download search index data (if not present) ===
echo "=== Step 1: Preparing search index data ==="
SEARCH_DATA_DIR="./search_data"
INDICES_DIR="${SEARCH_DATA_DIR}/prebuilt_indices"
LOCK_FILE="${SEARCH_DATA_DIR}/.data_prep.lock"

if [ -f "${INDICES_DIR}/corpus.json" ] && [ -f "${INDICES_DIR}/e5_Flat.index" ]; then
    echo "Search index data already exists, skipping download."
else
    # Use lock file to prevent concurrent downloads from multiple jobs
    echo "Acquiring data preparation lock..."
    mkdir -p "${SEARCH_DATA_DIR}"
    exec 9>"${LOCK_FILE}"
    flock -x 9

    # Re-check after acquiring lock (another job may have completed)
    if [ -f "${INDICES_DIR}/corpus.json" ] && [ -f "${INDICES_DIR}/e5_Flat.index" ]; then
        echo "Search index data prepared by another job, skipping."
    else
        echo "Downloading search index data..."
        python scripts/download_search_index.py --data_dir "${SEARCH_DATA_DIR}"

        # Post-processing: merge index shards into e5_Flat.index
        if [ -f "${INDICES_DIR}/part_aa" ] && [ -f "${INDICES_DIR}/part_ab" ] && [ ! -f "${INDICES_DIR}/e5_Flat.index" ]; then
            echo "Merging index shards: part_aa + part_ab -> e5_Flat.index"
            cat "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab" > "${INDICES_DIR}/e5_Flat.index"
            rm -f "${INDICES_DIR}/part_aa" "${INDICES_DIR}/part_ab"
            echo "Index merged."
        fi

        # Post-processing: convert wiki-18.jsonl to corpus.json
        if [ -f "${SEARCH_DATA_DIR}/wikipedia/wiki-18.jsonl" ] && [ ! -f "${INDICES_DIR}/corpus.json" ]; then
            echo "Converting wiki-18.jsonl -> corpus.json (this may take a few minutes)..."
            python -c "
import json
corpus = []
with open('${SEARCH_DATA_DIR}/wikipedia/wiki-18.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        corpus.append(doc['contents'])
with open('${INDICES_DIR}/corpus.json', 'w') as f:
    json.dump(corpus, f)
print(f'Created corpus.json with {len(corpus)} documents')
"
        fi

        echo "Search index data preparation complete."
    fi

    flock -u 9
    exec 9>&-
fi

# === Step 2: Prepare HotpotQA parquet data (if not present) ===
echo "=== Step 2: Preparing HotpotQA parquet data ==="
python scripts/prepare_search_data.py
echo "HotpotQA data preparation done."

# === Step 3: Start retrieval server in background ===
echo "=== Step 3: Starting retrieval server on port ${RETRIEVAL_PORT} ==="
python scripts/retrieval/server.py --data_dir ./search_data/prebuilt_indices --port "$RETRIEVAL_PORT" --host 0.0.0.0 &
RETRIEVAL_PID=$!
echo "Retrieval server started (PID=$RETRIEVAL_PID)"

# Health check: wait until server is ready
echo "Waiting for retrieval server health check..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${RETRIEVAL_PORT}/health" 2>/dev/null | grep -q "200"; then
        echo "Retrieval server is healthy (waited ${WAITED}s)."
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Retrieval server did not become healthy within ${MAX_WAIT}s"
    exit 1
fi

# Export retrieval URL so search env can find it
export RETRIEVAL_SERVER_URL="http://127.0.0.1:${RETRIEVAL_PORT}"
# Avoid NFS file lock issues: use node-local tmp for datasets cache
# Only redirect datasets cache, keep HF_HOME for auth tokens
export HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${SLURM_JOB_ID}"
mkdir -p "$HF_DATASETS_CACHE"
# === Step 4: Run single experiment ===
echo "=== Step 4: Running ${EXP_NAME} ==="
bash scripts/runs/run_search_benchmark.sh \
    --steps 200 \
    --models "$MODEL" \
    --algos "$ALGO" \
    --filters "$FILTER" \
    --gpus-per-exp "$NUM_GPUS" \
    --tp "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --micro-batch "$MICRO_BATCH" \
    --cooldown 0 \
    --retrieval-port "$RETRIEVAL_PORT"

echo ""
echo "=== Done ${EXP_NAME}: $(date) ==="
