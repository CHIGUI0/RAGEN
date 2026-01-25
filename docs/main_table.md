# Main Table Runs

This doc covers how to run `scripts/runs/run_main_table.sh`, which launches PPO/DAPO/GRPO/DrGRPO across multiple tasks with filter/no-filter settings.

## How to run
From the repo root:

```bash
bash scripts/runs/run_main_table.sh --steps 200 --tasks sokoban,webshop
```

Common options:
- `--model_name` (default: `Qwen2.5-3B`)
- `--steps` (default: `400`)
- `--tasks` (comma list; default: `sokoban,webshop,frozenlake,metamathqa,countdown`)
- `--gpus` (comma list; auto-detect if omitted)
- `--gpus-per-exp` (default: `1`; must divide GPU count)
- `--cooldown` (seconds between runs on the same GPU group; default: `30`)

Examples:
```bash
# Single task, quick sanity
bash scripts/runs/run_main_table.sh --steps 5 --tasks sokoban

# Two tasks, fixed GPUs
bash scripts/runs/run_main_table.sh --steps 200 --tasks sokoban,webshop --gpus 0,1,2,3

# Use 2 GPUs per experiment
bash scripts/runs/run_main_table.sh --steps 200 --gpus 0,1,2,3 --gpus-per-exp 2
```

## Outputs
- Per-task logs and results: `logs/main_table_<task>_<MODEL_NAME>/`
- Global summary log: `logs/main_table_<MODEL_NAME>.log`
