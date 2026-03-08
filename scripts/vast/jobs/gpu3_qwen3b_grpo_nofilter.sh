#!/bin/bash
# GPU 3 | Qwen2.5-3B-Instruct | GRPO | nofilter
cd "$(dirname "$0")/../../.." || exit 1
export MASTER_PORT=29503
bash scripts/vast/start_server_and_run.sh 8103 \
    bash scripts/runs/run_search_benchmark.sh \
    --models Qwen2.5-3B-Instruct \
    --algos GRPO \
    --filters nofilter \
    --gpus 3 \
    --gpus-per-exp 1 \
    --micro-batch 2 \
    --gpu-memory-utilization 0.4 \
    --collapse-freq 999 \
    --retrieval-port 8103
