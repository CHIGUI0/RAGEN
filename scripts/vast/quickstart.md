# Search Benchmark Quickstart (Vast.ai / Multi-GPU Server)

## Prerequisites

- NVIDIA GPU(s) with CUDA 12.x
- conda installed (script will install if missing)
- ~100GB free disk space (for wiki corpus + FAISS index)

## Step 1: Clone repo

```bash
git clone -b search-env https://github.com/lichengliu03/RAGEN.git
cd RAGEN
```

## Step 2: Install dependencies

```bash
bash scripts/vast/setup_vast.sh
```

This creates a `ragen` conda environment with:
- Python 3.12, PyTorch 2.5.0 (cu124), flash-attn 2.7.4
- vllm 0.8.2, transformers 4.48.2
- verl (submodule), ragen (editable install)
- Retrieval server deps (flask, sentence-transformers, faiss-cpu)
- wandb

Activate the environment:

```bash
conda activate ragen
```

## Step 3: Prepare data

```bash
bash scripts/vast/prepare_data.sh
```

Downloads and prepares:
- Wikipedia corpus + FAISS index (~74GB) → `search_data/prebuilt_indices/`
- HotpotQA train/val parquet → `data/search/`

Disk-intensive step. Index shards are deleted after merge to save space.

## Step 4: Start retrieval server

The retrieval server runs on **CPU** (no GPU needed), serving dense retrieval over Wikipedia.

```bash
python scripts/retrieval/server.py \
    --data_dir ./search_data/prebuilt_indices \
    --port 8000 --host 127.0.0.1 &
```

Wait for it to load (~2-5 min for 61GB index), then verify:

```bash
curl http://127.0.0.1:8000/health
# Should return: {"status":"healthy","corpus_size":21015324,...}
```

> Note: `run_all_search.sh` starts the server automatically. Only start it manually if running individual jobs.

## Step 5: (Optional) Diagnostic inference

Before training, verify the model gets meaningful reward variance across rollouts:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vast/run_search_inference.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --n_prompts 30 \
    --rollouts_per_prompt 8 \
    --temperature 0.7 \
    --max_turns 5 \
    --retrieval_port 8000 \
    --output logs/search_inference.json
```

Then visualize the reward matrix:

```bash
python scripts/vast/plot_reward_matrix.py --input logs/search_inference.json
```

**What to look for:**
- Mixed (learnable) prompts > 20% → RL has signal, proceed to training
- All-wrong > 50% → check retrieval server, prompt format, model capability
- All-correct > 50% → task too easy, consider harder subset

## Step 6: Login to wandb

```bash
wandb login <your_api_key>
```

Training logs to project `ragen_search_benchmark`.

## Step 7: Run training

### Single experiment

```bash
# Ensure retrieval server is running (Step 4)
conda activate ragen
bash scripts/vast/jobs/gpu0_qwen3b_ppo_filter.sh
```

Available job scripts in `scripts/vast/jobs/`:

| Script | Model | Algo | Filter | GPU |
|--------|-------|------|--------|-----|
| `gpu0_qwen3b_ppo_filter.sh` | Qwen2.5-3B | PPO | filter | 0 |
| `gpu1_qwen3b_ppo_nofilter.sh` | Qwen2.5-3B | PPO | nofilter | 1 |
| `gpu2_qwen3b_grpo_filter.sh` | Qwen2.5-3B | GRPO | filter | 2 |
| `gpu3_qwen3b_grpo_nofilter.sh` | Qwen2.5-3B | GRPO | nofilter | 3 |
| `gpu4_llama3b_ppo_filter.sh` | Llama-3.2-3B | PPO | filter | 4 |
| `gpu5_llama3b_ppo_nofilter.sh` | Llama-3.2-3B | PPO | nofilter | 5 |
| `gpu01_qwen7b_ppo_filter.sh` | Qwen2.5-7B | PPO | filter | 0,1 |
| `gpu23_qwen7b_ppo_nofilter.sh` | Qwen2.5-7B | PPO | nofilter | 2,3 |

### All experiments (automated)

```bash
# Starts retrieval server + runs Phase 1 (3B models) then Phase 2 (7B models)
bash scripts/vast/run_all_search.sh
```

## Monitoring

```bash
# Training logs
tail -f logs/search_benchmark/search-PPO-filter-Qwen2.5-3B-Instruct.log

# GPU usage
watch nvidia-smi

# Retrieval server
tail -f logs/retrieval_server.log

# wandb dashboard
# https://wandb.ai/<your-entity>/ragen_search_benchmark
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'hydra'` | `pip install hydra-core` |
| `No module named 'pkg_resources'` | `pip install "setuptools<70.0.0"` |
| webshop submodule SSH error | `git submodule update --init verl` (skip webshop) |
| Disk full during index merge | Clear HF cache: `rm -rf ~/.cache/huggingface/hub` |
| wandb 401 error | Re-run `wandb login` with full key (including `wandb_v1_` prefix) |
| Prompt too long crash in inference | Already handled with truncation check in `run_search_inference.py` |
