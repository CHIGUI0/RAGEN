import wandb
import matplotlib.pyplot as plt
import os

# CONFIGURATION
WANDB_PATH = "deimos-xing/AGEN_gradient_analysis/jkvkalpa"
BUCKETS = [
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
    "bucket_5",
    "bucket_6",
    "bucket_7",
    "bucket_8",
]
COMPONENTS = ["kl", "entropy", "task"]
LOSS_COMPONENTS = ["policy", "entropy", "kl", "total"]
OUTPUT_DIR = "gradient_plots"
TARGET_STEPS = "all"  # "all", list like [1, 11], or None to auto-pick lowest step with bucket data

def get_bucket_label(bucket_name):
    """Formats bucket names for the plot axis."""
    if bucket_name.startswith("bucket_"):
        return bucket_name.replace("_", " ")
    return bucket_name

def main():
    print(f"Connecting to WandB run: {WANDB_PATH}...")
    api = wandb.Api()
    try:
        run = api.run(WANDB_PATH)
    except Exception as e:
        print(f"Error accessing run: {e}")
        return

    summary = run.summary
    step_metrics = {}
    available_steps = set()
    for row in run.scan_history():
        bucket_items = [(k, v) for k, v in row.items() if k.startswith("grad_norm/bucket_")]
        has_bucket_metrics = any(v is not None for _, v in bucket_items)
        has_nonzero_bucket_metrics = any((v is not None and v != 0) for _, v in bucket_items)
        if not has_bucket_metrics or not has_nonzero_bucket_metrics:
            continue
        step = row.get("_step")
        if step is None:
            continue
        available_steps.add(step)
        if step not in step_metrics:
            step_metrics[step] = {}
        for k, v in row.items():
            if v is None:
                continue
            step_metrics[step][k] = v

    if available_steps:
        print(f"Found grad_norm bucket metrics at steps: {sorted(available_steps)}")
    else:
        print("Warning: no grad_norm metrics found in history; falling back to run summary.")
        step_metrics = {"summary": summary}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    x_labels = [get_bucket_label(b) for b in BUCKETS]

    titles = {
        "kl": "KL Gradient Norm",
        "entropy": "Entropy Gradient Norm",
        "task": "Task (Policy) Gradient Norm"
    }
    
    colors = ["#3498db", "#2ecc71", "#e74c3c"] # Blue, Green, Red
    loss_titles = {
        "policy": "Policy (Task) Loss",
        "entropy": "Entropy Loss",
        "kl": "KL Loss",
        "total": "Total Loss",
    }
    loss_colors = ["#8e44ad", "#27ae60", "#2980b9", "#c0392b"]  # Purple, Green, Blue, Red
    norm_titles = {
        "kl": "KL Grad Norm (Per Sample vs Per Token)",
        "entropy": "Entropy Grad Norm (Per Sample vs Per Token)",
        "task": "Task Grad Norm (Per Sample vs Per Token)",
    }

    steps_to_plot = sorted(step_metrics.keys(), key=lambda x: (isinstance(x, str), x))
    if TARGET_STEPS == "all":
        pass
    elif TARGET_STEPS:
        steps_to_plot = [s for s in steps_to_plot if s in TARGET_STEPS]
    elif steps_to_plot and steps_to_plot[0] != "summary":
        steps_to_plot = [steps_to_plot[0]]

    for step_key in steps_to_plot:
        metric_source = step_metrics[step_key]
        step_tag = f"step_{step_key}"
        output_file = os.path.join(OUTPUT_DIR, f"gradient_analysis_plots_{step_tag}.png")
        output_file_loss = os.path.join(OUTPUT_DIR, f"gradient_analysis_loss_plots_{step_tag}.png")
        output_file_rv = os.path.join(OUTPUT_DIR, f"gradient_analysis_reward_std_{step_tag}.png")
        output_file_normed = os.path.join(OUTPUT_DIR, f"gradient_analysis_normed_grads_{step_tag}.png")

        # Create subplots for gradient norms
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.3, top=0.62, bottom=0.12)

        bucket_rv = {
            b: {
                "mean": metric_source.get(f"grad_norm/{b}/reward_std_mean", 0),
                "min": metric_source.get(f"grad_norm/{b}/reward_std_min", 0),
                "max": metric_source.get(f"grad_norm/{b}/reward_std_max", 0),
            }
            for b in BUCKETS
        }
        legend_lines = []
        for label, bucket in zip(x_labels, BUCKETS):
            rv = bucket_rv.get(bucket, {})
            legend_lines.append(
                f"{label}: mean={rv.get('mean', 0):.3f} min={rv.get('min', 0):.3f} max={rv.get('max', 0):.3f}"
            )
        for ax, comp, color in zip(axes, COMPONENTS, colors):
            y_values = []
            for bucket in BUCKETS:
                key = f"grad_norm/{bucket}/{comp}"
                val = metric_source.get(key, 0)
                y_values.append(val)

            bars = ax.bar(x_labels, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title(titles[comp], fontsize=16, fontweight='bold', pad=15)
            ax.set_ylabel("Grad Norm Magnitude", fontsize=12)
            ax.set_xlabel("Reward Variance Bucket", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(y_values)*0.01 if y_values else 0.01),
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle(f"Gradient Norms - Run: {run.name} (Step {step_key})", fontsize=20, y=0.98)
        fig.text(0.5, 0.88, "\n".join(legend_lines), ha="center", va="top", fontsize=8)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\nSuccess! Results visualization saved to: {os.path.abspath(output_file)}")
        plt.close(fig)

        # Create subplots for per-component losses
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for ax, comp, color in zip(axes2.flatten(), LOSS_COMPONENTS, loss_colors):
            y_values = []
            for bucket in BUCKETS:
                key = f"grad_norm/{bucket}/loss/{comp}"
                val = metric_source.get(key, 0)
                y_values.append(val)

            bars = ax.bar(x_labels, y_values, color=color, alpha=0.8, edgecolor="black", linewidth=1)
            ax.set_title(loss_titles[comp], fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(y_values) * 0.01 if y_values else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        fig2.suptitle(f"Per-Component Losses - Run: {run.name} (Step {step_key})", fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(output_file_loss, bbox_inches="tight", dpi=300)
        print(f"Success! Loss visualization saved to: {os.path.abspath(output_file_loss)}")
        plt.close(fig2)

    # Create plot for per-bucket mean reward variance (std)
        rv_values = []
        for bucket in BUCKETS:
            rv_values.append(metric_source.get(f"grad_norm/{bucket}/reward_std_mean", 0))
        if any(v != 0 for v in rv_values):
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
            bars = ax3.bar(x_labels, rv_values, color="#f39c12", alpha=0.85, edgecolor="black", linewidth=1)
            ax3.set_title(f"Reward Std Mean by Bucket - Run: {run.name} (Step {step_key})", fontsize=14, fontweight="bold", pad=10)
            ax3.set_ylabel("Reward Std (Mean)", fontsize=11)
            ax3.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax3.grid(axis="y", linestyle="--", alpha=0.6)
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(rv_values) * 0.01 if rv_values else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            plt.tight_layout()
            plt.savefig(output_file_rv, bbox_inches="tight", dpi=300)
            print(f"Success! Reward std visualization saved to: {os.path.abspath(output_file_rv)}")
            plt.close(fig3)
        else:
            print(f"Warning: No reward std mean metrics found at step {step_key}; skipping reward std plot.")

        # Create plots for per-sample and per-token grad norms (combined per component)
        fig4, axes4 = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.3)
        for ax, comp in zip(axes4, COMPONENTS):
            per_sample = []
            per_token = []
            for bucket in BUCKETS:
                per_sample.append(metric_source.get(f"grad_norm/{bucket}/per_sample/{comp}", 0))
                per_token.append(metric_source.get(f"grad_norm/{bucket}/per_token/{comp}", 0))

            x = range(len(x_labels))
            width = 0.38
            bars1 = ax.bar([i - width / 2 for i in x], per_sample, width=width, label="per_sample", color="#16a085", alpha=0.85)
            bars2 = ax.bar([i + width / 2 for i in x], per_token, width=width, label="per_token", color="#f39c12", alpha=0.85)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels)
            ax.set_title(norm_titles[comp], fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Grad Norm", fontsize=11)
            ax.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend(frameon=False, fontsize=9)

            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(per_sample + per_token) * 0.01 if (per_sample + per_token) else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig4.suptitle(f"Normalized Grad Norms - Run: {run.name} (Step {step_key})", fontsize=18, y=1.03)
        plt.tight_layout()
        plt.savefig(output_file_normed, bbox_inches="tight", dpi=300)
        print(f"Success! Normalized grad visualization saved to: {os.path.abspath(output_file_normed)}")
        plt.close(fig4)

if __name__ == "__main__":
    main()
