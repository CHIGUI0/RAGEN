#!/bin/bash
# GPU 4 | Llama-3.2-3B-Instruct | PPO | filter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29504
bash scripts/vast/start_server_and_run.sh 8104 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Llama-3.2-3B-Instruct \
    --algos PPO \
    --filters filter \
    --gpus 4 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8104
