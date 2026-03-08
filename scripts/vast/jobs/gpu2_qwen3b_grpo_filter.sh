#!/bin/bash
# GPU 2 | Qwen2.5-3B-Instruct | GRPO | filter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29502
bash scripts/vast/start_server_and_run.sh 8102 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-3B-Instruct \
    --algos GRPO \
    --filters filter \
    --gpus 2 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8102
