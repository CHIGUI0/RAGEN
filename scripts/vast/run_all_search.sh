#!/bin/bash
# Run all Search benchmark experiments on Vast.ai 8×H200
#
# Shared retrieval server (port 8000) + two phases:
#   Phase 1: 6 × 3B experiments (1 GPU each, parallel)
#     - Qwen2.5-3B-Instruct: PPO + GRPO, filter + nofilter  (4 exps, GPUs 0-3)
#     - Llama-3.2-3B-Instruct: PPO, filter + nofilter        (2 exps, GPUs 4-5)
#   Phase 2: 2 × 7B experiments (2 GPUs each, parallel)
#     - Qwen2.5-7B-Instruct: PPO, filter + nofilter          (2 exps, GPUs 0-3)

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_step() { echo -e "${BLUE}[$(date '+%H:%M:%S')] ${1}${NC}"; }
print_ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ${1}${NC}"; }
print_err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ${1}${NC}"; }

if [ ! -f "train.py" ]; then
    echo "Please run this script from the RAGEN repo root."
    exit 1
fi

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate ragen

RETRIEVAL_PORT=8000
DATA_DIR="./search_data/prebuilt_indices"
SERVER_LOG="logs/retrieval_server.log"
mkdir -p logs/search_benchmark

# ============================================================
# Helper: cleanup retrieval server on exit
# ============================================================
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        print_step "Stopping retrieval server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        print_ok "Retrieval server stopped"
    fi
}
trap cleanup EXIT

# ============================================================
# 1. Start shared retrieval server
# ============================================================
print_step "Starting retrieval server on port ${RETRIEVAL_PORT}..."

python scripts/retrieval/server.py \
    --data_dir "$DATA_DIR" \
    --port "$RETRIEVAL_PORT" \
    --host 127.0.0.1 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

print_step "Waiting for retrieval server (PID $SERVER_PID) to become healthy..."
MAX_WAIT=300  # 5 minutes (loading ~74GB index takes time)
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        print_err "Retrieval server process died. Check $SERVER_LOG"
        tail -20 "$SERVER_LOG"
        exit 1
    fi
    if curl -s "http://127.0.0.1:${RETRIEVAL_PORT}/health" | grep -q '"status"'; then
        print_ok "Retrieval server is healthy!"
        curl -s "http://127.0.0.1:${RETRIEVAL_PORT}/health" | python3 -m json.tool
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if (( WAITED % 30 == 0 )); then
        print_step "Still waiting... (${WAITED}s / ${MAX_WAIT}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    print_err "Retrieval server did not become healthy within ${MAX_WAIT}s"
    tail -20 "$SERVER_LOG"
    exit 1
fi

# ============================================================
# 2. Phase 1 — 6 × 3B experiments (1 GPU each)
# ============================================================
print_step "========== Phase 1: 3B models (6 experiments, 1 GPU each) =========="

# Phase 1a: Qwen2.5-3B-Instruct — PPO + GRPO × {filter, nofilter} = 4 experiments
# Phase 1b: Llama-3.2-3B-Instruct — PPO × {filter, nofilter} = 2 experiments
# Run both in parallel, split GPUs to avoid contention

print_step "Phase 1a: Qwen2.5-3B-Instruct (PPO+GRPO) on GPUs 0,1,2,3..."
bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-3B-Instruct \
    --algos PPO,GRPO \
    --filters all \
    --gpus 0,1,2,3 \
    --gpus-per-exp 1 \
    --retrieval-port "$RETRIEVAL_PORT" \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    2>&1 | tee logs/search_benchmark/phase1a_qwen3b.log &
PID_PHASE1A=$!

print_step "Phase 1b: Llama-3.2-3B-Instruct (PPO) on GPUs 4,5..."
bash scripts/runs/run_search_benchmark.sh \
    --models Llama-3.2-3B-Instruct \
    --algos PPO \
    --filters all \
    --gpus 4,5 \
    --gpus-per-exp 1 \
    --retrieval-port "$RETRIEVAL_PORT" \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    2>&1 | tee logs/search_benchmark/phase1b_llama3b.log &
PID_PHASE1B=$!

# Wait for both Phase 1 sub-tasks
PHASE1_OK=true
wait "$PID_PHASE1A" || { print_err "Phase 1a (Qwen 3B) failed"; PHASE1_OK=false; }
wait "$PID_PHASE1B" || { print_err "Phase 1b (Llama 3B) failed"; PHASE1_OK=false; }

if [ "$PHASE1_OK" = true ]; then
    print_ok "Phase 1 complete!"
else
    print_err "Phase 1 had failures (continuing to Phase 2)"
fi

# ============================================================
# 3. Phase 2 — 2 × 7B experiments (2 GPUs each)
# ============================================================
print_step "========== Phase 2: 7B model (2 experiments, 2 GPUs each) =========="

bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-7B-Instruct \
    --algos PPO \
    --filters all \
    --gpus 0,1,2,3 \
    --gpus-per-exp 2 \
    --retrieval-port "$RETRIEVAL_PORT" \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    2>&1 | tee logs/search_benchmark/phase2_qwen7b.log
PHASE2_EXIT=$?

if [ $PHASE2_EXIT -eq 0 ]; then
    print_ok "Phase 2 complete!"
else
    print_err "Phase 2 failed (exit code $PHASE2_EXIT)"
fi

# ============================================================
# 4. Summary
# ============================================================
echo ""
echo "========================================================"
echo "  Search Benchmark — All Phases Complete"
echo "========================================================"
echo ""

print_step "Individual experiment results:"
for f in logs/search_benchmark/*.result; do
    [ -f "$f" ] && cat "$f"
done

echo ""
print_step "Phase logs:"
echo "  Phase 1a (Qwen 3B):  logs/search_benchmark/phase1a_qwen3b.log"
echo "  Phase 1b (Llama 3B): logs/search_benchmark/phase1b_llama3b.log"
echo "  Phase 2  (Qwen 7B):  logs/search_benchmark/phase2_qwen7b.log"
echo "  Retrieval server:    $SERVER_LOG"
echo ""
print_ok "Done! Check wandb for training curves."
