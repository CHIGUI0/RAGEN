#!/bin/bash
# GPU 5 | Llama-3.2-3B-Instruct | PPO | nofilter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29505
bash scripts/vast/start_server_and_run.sh 8105 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Llama-3.2-3B-Instruct \
    --algos PPO \
    --filters nofilter \
    --gpus 5 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8105
