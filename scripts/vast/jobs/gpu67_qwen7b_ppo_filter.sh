#!/bin/bash
# GPU 6,7 | Qwen2.5-7B-Instruct | PPO | filter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29509
bash scripts/vast/start_server_and_run.sh 8109 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-7B-Instruct \
    --algos PPO \
    --filters filter \
    --gpus 6,7 \
    --gpus-per-exp 2 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8109
