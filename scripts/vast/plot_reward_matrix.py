#!/usr/bin/env python3
"""
Plot the reward triangular matrix from multi-rollout inference results.

Reads the JSON output from run_search_inference.py and generates a heatmap where:
  - Each row = one prompt, rollout rewards sorted high-to-low within the row
  - Rows sorted by Reward Variance (RV) from high (top) to low (bottom)
  - RV annotated on the right y-axis
  - Color: reward value (0 = red/dark, 1 = green/bright)

The resulting "triangular" shape shows:
  - Top rows: high RV (mixed rewards) = learnable prompts
  - Bottom rows with all 1s: too easy
  - Bottom rows with all 0s: too hard or broken

Usage:
    python scripts/vast/plot_reward_matrix.py \
        --input logs/search_inference.json \
        --output logs/reward_matrix.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot reward triangular matrix")
    parser.add_argument("--input", required=True, help="Path to inference JSON from run_search_inference.py")
    parser.add_argument("--output", default=None, help="Output image path (default: <input_stem>_matrix.png)")
    parser.add_argument("--max_prompts", type=int, default=100, help="Max prompts to display (for readability)")
    parser.add_argument("--figsize_w", type=float, default=12, help="Figure width in inches")
    parser.add_argument("--figsize_h", type=float, default=None, help="Figure height (auto if not set)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    config = data["config"]
    prompts = data["prompts"]
    n_rollouts = config["rollouts_per_prompt"]

    # Build matrix: each row = sorted rewards (descending) for one prompt
    reward_rows = []
    rv_values = []
    questions = []

    for p in prompts:
        rewards = sorted(p["rewards"], reverse=True)  # high to low within row
        reward_rows.append(rewards)
        rv_values.append(p["reward_variance"])
        q = p["question"]
        questions.append(q[:50] + "..." if len(q) > 50 else q)

    # Sort rows by RV descending (high variance at top)
    sort_idx = np.argsort(rv_values)[::-1]
    reward_rows = [reward_rows[i] for i in sort_idx]
    rv_values = [rv_values[i] for i in sort_idx]
    questions = [questions[i] for i in sort_idx]

    # Truncate for display
    n_display = min(len(reward_rows), args.max_prompts)
    reward_rows = reward_rows[:n_display]
    rv_values = rv_values[:n_display]
    questions = questions[:n_display]

    matrix = np.array(reward_rows)  # shape: (n_prompts, n_rollouts)

    # Compute summary regions
    n_mixed = sum(1 for rv in rv_values if rv > 0)
    n_easy = sum(1 for i, rv in enumerate(rv_values) if rv == 0 and all(r == 1.0 for r in reward_rows[i]))
    n_hard = sum(1 for i, rv in enumerate(rv_values) if rv == 0 and all(r == 0.0 for r in reward_rows[i]))

    # Figure sizing
    if args.figsize_h is None:
        figsize_h = max(6, n_display * 0.25)
    else:
        figsize_h = args.figsize_h

    fig, ax = plt.subplots(figsize=(args.figsize_w, figsize_h))

    # Custom colormap: dark red (0) -> yellow (0.5) -> green (1.0)
    cmap = LinearSegmentedColormap.from_list(
        "reward",
        [(0.0, "#d32f2f"), (0.3, "#ff9800"), (0.5, "#ffeb3b"), (0.7, "#8bc34a"), (1.0, "#2e7d32")],
    )

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    # X axis: rollout index
    ax.set_xlabel(f"Rollouts (sorted high→low within each prompt)", fontsize=11)
    ax.set_xticks(range(n_rollouts))
    ax.set_xticklabels([f"R{i}" for i in range(n_rollouts)], fontsize=8)

    # Y axis left: prompt index
    ax.set_ylabel("Prompts (sorted by RV, high→low)", fontsize=11)
    if n_display <= 60:
        ax.set_yticks(range(n_display))
        ax.set_yticklabels([f"{i}" for i in range(n_display)], fontsize=6)
    else:
        tick_step = max(1, n_display // 30)
        ax.set_yticks(range(0, n_display, tick_step))

    # Y axis right: RV values
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    if n_display <= 60:
        ax2.set_yticks(range(n_display))
        ax2.set_yticklabels([f"RV={rv:.3f}" for rv in rv_values], fontsize=6)
    else:
        # Show RV at key positions
        key_positions = list(range(0, n_display, max(1, n_display // 20)))
        ax2.set_yticks(key_positions)
        ax2.set_yticklabels([f"RV={rv_values[i]:.3f}" for i in key_positions], fontsize=6)

    # Colorbar
    cbar = fig.colorbar(im, ax=[ax, ax2], pad=0.12, shrink=0.8)
    cbar.set_label("Reward", fontsize=10)

    # Title with summary
    title = (
        f"Reward Matrix: {config['model']} | "
        f"{n_display} prompts x {n_rollouts} rollouts | temp={config['temperature']}\n"
        f"Mixed (learnable): {n_mixed} | All-correct (easy): {n_easy} | All-wrong (hard/broken): {n_hard}"
    )
    ax.set_title(title, fontsize=12, pad=12)

    # Draw horizontal separator lines between regions
    if n_mixed > 0 and n_mixed < n_display:
        ax.axhline(y=n_mixed - 0.5, color="white", linewidth=2, linestyle="--")
        ax.text(n_rollouts + 0.3, n_mixed - 0.5, "← RV=0 below", fontsize=7,
                va="center", color="gray", transform=ax.transData)

    fig.subplots_adjust(left=0.06, right=0.82, top=0.90, bottom=0.10)

    # Output path
    if args.output is None:
        output_path = Path(args.input).with_name(Path(args.input).stem + "_matrix.png")
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved reward matrix to {output_path}")

    # Also print text summary table
    print(f"\n{'=' * 70}")
    print(f"REWARD MATRIX SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model:           {config['model']}")
    print(f"Prompts:         {n_display}")
    print(f"Rollouts/prompt: {n_rollouts}")
    print(f"Temperature:     {config['temperature']}")
    print(f"")
    print(f"Mixed (RV > 0):  {n_mixed:4d}  ({n_mixed/n_display*100:5.1f}%)  ← RL can learn from these")
    print(f"All correct:     {n_easy:4d}  ({n_easy/n_display*100:5.1f}%)  ← too easy, no signal")
    print(f"All wrong:       {n_hard:4d}  ({n_hard/n_display*100:5.1f}%)  ← too hard or broken")
    print(f"")

    mean_rv_mixed = np.mean([rv for rv in rv_values if rv > 0]) if n_mixed > 0 else 0
    print(f"Mean RV (mixed only): {mean_rv_mixed:.4f}")
    print(f"Overall mean reward:  {data['summary']['mean_reward']:.4f}")
    print(f"")

    if n_hard / max(n_display, 1) > 0.5:
        print("DIAGNOSIS: >50% prompts are all-wrong.")
        print("  -> Check: Is the retrieval server running and returning good results?")
        print("  -> Check: Does the model understand the search[]/finish[] format?")
        print("  -> Check: Is the prompt/system instruction clear?")
    elif n_easy / max(n_display, 1) > 0.5:
        print("DIAGNOSIS: >50% prompts are all-correct.")
        print("  -> Task may be too easy for this model. Consider harder subset or lower temperature.")
    elif n_mixed / max(n_display, 1) < 0.2:
        print("DIAGNOSIS: <20% prompts have mixed rewards. Weak RL signal.")
        print("  -> Adjust temperature, check retrieval quality, or use different data subset.")
    else:
        print(f"DIAGNOSIS: Good RL signal. {n_mixed/n_display*100:.0f}% prompts are learnable.")
        print("  -> Proceed to 50-step smoke test.")

    plt.close(fig)


if __name__ == "__main__":
    main()
