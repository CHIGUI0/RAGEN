#!/bin/bash
# GPU 0 | Qwen2.5-3B-Instruct | PPO | filter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29500
bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-3B-Instruct \
    --algos PPO \
    --filters filter \
    --gpus 0 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8000
