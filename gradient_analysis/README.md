# Gradient Analysis (ICML Plots)

This folder is self‑contained: it holds the metrics needed to reproduce the ICML plots and the plotting script.

## Contents
- `data/grpo/step{0,20,40}/metrics.json`
- `data/ppo/step{0,20,40}/metrics.json`
- `plot_icml_steps.py`

## Generate plots

From repo root, using the existing venv:

```bash
./.venv/bin/python gradient_analysis/plot_icml_steps.py \
  --mode grpo --out gradient_analysis/icml_grpo_step0_20_40_grid.png

./.venv/bin/python gradient_analysis/plot_icml_steps.py \
  --mode ppo --out gradient_analysis/icml_ppo_step0_20_40_grid.png
```

## Override data paths (optional)

You can override any step directory to point to different metrics:

```bash
./.venv/bin/python gradient_analysis/plot_icml_steps.py \
  --mode grpo \
  --step0-dir /path/to/step0 \
  --step20-dir /path/to/step20 \
  --step40-dir /path/to/step40 \
  --out /path/to/out.png
```

Each step directory must contain `metrics.json`.
