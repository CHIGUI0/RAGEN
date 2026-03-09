# Reward Variance Heatmap Analysis

## Overview
This directory contains a reproducible Python script for generating Reward Variance Heatmaps dynamically from W&B logging artifacts during execution. 

In `ragen/trainer/agent_trainer.py`, the execution framework stores tabular W&B artifacts containing the sampled outcomes of each active step. The Python script within `scripts/plot_wandb_heatmaps.py` fetches these artifacts locally, calculates the corresponding Standard Deviations for each group across its samples, evaluates the `linear` Top-P preservation threshold, and visualizes the distribution using Matplotlib and seaborn Heatmaps.

## Top-P Filtering Verification
The `linear` mode under `top_p=0.9` operates by keeping groups sequentially until the accumulated variance explicitly surpasses `0.9 * Total System Variance`.

As verified by this tracking metric extraction:
- The expected fixed 90% preservation translates to dynamically retaining anywhere between 4 and 7 groups out of a total set of 8.
- Evaluation matches **100.0%** correlation between simulated filter_kept_ratios on Group-level Standard Deviations against the backend `rollout/filter_kept_ratio` mapped to W&B.

## Visualizations
The visual components available natively track the selection procedure over variable steps:
- **Y-Axis**: Groups sorted downwards by descending Standard Deviation parameter bounds.
- **X-Axis**: Ordered sample distributions (Lowest to Highest Reward score).
- **Blue Selection Line**: The explicitly tracked cutoff defining whether a subset falls inside the preservation domain or drops.

### Usage
Generate the sequence natively in via `python`:

```bash
python scripts/plot_wandb_heatmaps.py \
    --run "deimos-xing/main_webshop/h3v7xc1r" \
    --steps 0,10,20,30,40 \
    --top_p 0.9 \
    --out plots/
```

Reference pre-generated instances under the `plots/` folder covering the W&B run progression structure.
