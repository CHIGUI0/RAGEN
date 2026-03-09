# Gradient Analysis Plotting

This folder contains the plotting script used for the gradient-analysis figures.

The repository does not ship the underlying `metrics.json` files. Point the script at
directories produced by your own runs.

## Contents
- `plot_icml_steps.py`

## Generate plots

From repo root:

```bash
python gradient_analysis/plot_icml_steps.py \
  --mode grpo \
  --step0-dir /path/to/step0 \
  --step20-dir /path/to/step20 \
  --step40-dir /path/to/step40 \
  --out /path/to/grpo.png

python gradient_analysis/plot_icml_steps.py \
  --mode ppo \
  --step0-dir /path/to/step0 \
  --step20-dir /path/to/step20 \
  --step40-dir /path/to/step40 \
  --out /path/to/ppo.png
```

Each step directory must contain `metrics.json`.
