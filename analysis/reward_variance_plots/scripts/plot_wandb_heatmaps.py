import argparse
import wandb
import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_top_p_linear(stds, top_p, eps=0.01):
    scores = np.sort(stds)[::-1]
    threshold = top_p * np.sum(scores) - eps
    cumulative_score = 0.0
    selected_count = 0
    for score in scores:
        if cumulative_score >= threshold: break
        if score <= 0: break
        cumulative_score += score
        selected_count += 1
    return selected_count if cumulative_score >= threshold else 0

def get_version(name):
    try:
        if ':v' in name: return int(name.split(':v')[-1])
        return 0
    except: return 0

def fetch_table_and_plot(run_path, step, top_p=0.9, out_dir="plots"):
    api = wandb.Api()
    run = api.run(run_path)
    artifacts = run.logged_artifacts()
    reward_table_artifacts = [a for a in artifacts if "reward_table" in a.name or "table" in a.name]
    reward_table_artifacts.sort(key=lambda x: get_version(x.name))
    
    target_artifact = next((a for a in reward_table_artifacts if get_version(a.name) == step), None)
    if not target_artifact:
        print(f"Artifact for step {step} not found! Skipping.")
        return False

    dl_path = f"/tmp/wandb_artifacts/{run.id}/step_{step}"
    if not os.path.exists(dl_path):
        target_artifact.download(root=dl_path)

    table_files = []
    for r, d, f in os.walk(dl_path):
        for file in f:
            if file.endswith('.table.json'):
                table_files.append(os.path.join(r, file))

    if not table_files:
        print(f"No json table found in {dl_path}. Skipping.")
        return False

    with open(table_files[0], 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data['data'], columns=data['columns'])

    stds = df.std(ddof=1)
    sorted_groups = stds.sort_values(ascending=False).index.tolist()
    df_sorted = df[sorted_groups]
    preserved_count = simulate_top_p_linear(stds.values, top_p)

    heatmap_data = df_sorted.T
    heatmap_values = np.sort(heatmap_data.values, axis=1)[:, ::-1]

    plt.figure(figsize=(10, 6))
    cmap = sns.color_palette("Reds", as_cmap=True)

    ax = sns.heatmap(
        heatmap_values, 
        cmap=cmap, 
        cbar_kws={'label': 'Reward Value'},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_yticks(np.arange(len(heatmap_data.index)) + 0.5)
    ax.set_yticklabels(heatmap_data.index)

    if preserved_count > 0 and preserved_count < len(sorted_groups):
        ax.axhline(preserved_count, color='blue', lw=3, label=f"Cutoff (Top {preserved_count} Kept)")
        
    plt.title(f"Step {step}: Rollout Reward Selection (Linear top-p={top_p})\nTop {preserved_count}/{len(sorted_groups)} groups kept")
    plt.xlabel("Sample Instances (Highest -> Lowest)")
    plt.ylabel("Groups (Highest Std -> Lowest Std)")
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"reward_heatmap_step_{step}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and plot reward variance heatmaps from W&B.")
    parser.add_argument("--run", type=str, required=True, help="W&B run path (e.g. deimos-xing/main_webshop/h3v7xc1r)")
    parser.add_argument("--steps", type=str, required=True, help="Comma-separated steps to plot (e.g. 0,10,20)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Linear top-p cutoff parameter")
    parser.add_argument("--out", type=str, default="plots", help="Output directory for PNG heatmaps")
    
    args = parser.parse_args()
    
    steps = [int(s.strip()) for s in args.steps.split(',')]
    for step in steps:
        print(f"Processing step {step}...")
        fetch_table_and_plot(args.run, step, args.top_p, args.out)

