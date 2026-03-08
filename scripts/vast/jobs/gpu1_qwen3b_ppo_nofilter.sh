#!/bin/bash
# GPU 1 | Qwen2.5-3B-Instruct | PPO | nofilter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29501
bash scripts/vast/start_server_and_run.sh 8101 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-3B-Instruct \
    --algos PPO \
    --filters nofilter \
    --gpus 1 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8101
