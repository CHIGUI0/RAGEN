#!/bin/bash
# Wrapper: start a per-experiment retrieval server, run training, then clean up.
# Usage: bash scripts/vast/start_server_and_run.sh <port> <run_args...>
#
# Example:
#   bash scripts/vast/start_server_and_run.sh 8000 \
#       scripts/runs/run_search_benchmark.sh --models Qwen2.5-3B-Instruct ...

set -euo pipefail

RETRIEVAL_PORT="$1"
shift

cd "$(dirname "$0")/../.." || exit 1

DATA_DIR="./search_data/prebuilt_indices"
SERVER_LOG="logs/retrieval_server_${RETRIEVAL_PORT}.log"
mkdir -p logs

# Cleanup on exit
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] Stopping retrieval server (PID=$SERVER_PID, port=$RETRIEVAL_PORT)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] Retrieval server stopped."
    fi
}
trap cleanup EXIT

# Start retrieval server
echo "[$(date '+%H:%M:%S')] Starting retrieval server on port ${RETRIEVAL_PORT}..."
python scripts/retrieval/server.py \
    --data_dir "$DATA_DIR" \
    --port "$RETRIEVAL_PORT" \
    --host 127.0.0.1 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Health check
echo "[$(date '+%H:%M:%S')] Waiting for retrieval server (PID=$SERVER_PID) to become healthy..."
MAX_WAIT=300
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] ERROR: Retrieval server died. Check $SERVER_LOG"
        tail -20 "$SERVER_LOG"
        exit 1
    fi
    if curl -s "http://127.0.0.1:${RETRIEVAL_PORT}/health" | grep -q '"status"'; then
        echo "[$(date '+%H:%M:%S')] Retrieval server is healthy (waited ${WAITED}s)."
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "[$(date '+%H:%M:%S')] ERROR: Retrieval server not healthy within ${MAX_WAIT}s"
    tail -20 "$SERVER_LOG"
    exit 1
fi

# Run the training command
echo "[$(date '+%H:%M:%S')] Running: $*"
"$@"
