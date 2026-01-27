import wandb
import matplotlib.pyplot as plt
import os

# CONFIGURATION
WANDB_PATH = "deimos-xing/AGEN_gradient_analysis/ik2b182p"
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
OUTPUT_FILE = "gradient_analysis_plots.png"
OUTPUT_FILE_LOSS = "gradient_analysis_loss_plots.png"
OUTPUT_FILE_RV = "gradient_analysis_reward_std.png"
OUTPUT_FILE_NORMED = "gradient_analysis_normed_grads.png"

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
    x_labels = [get_bucket_label(b) for b in BUCKETS]
    
    # Create subplots for gradient norms
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)

    titles = {
        "kl": "KL Gradient Norm",
        "entropy": "Entropy Gradient Norm",
        "task": "Task (Policy) Gradient Norm"
    }
    
    colors = ["#3498db", "#2ecc71", "#e74c3c"] # Blue, Green, Red

    bucket_rv = {b: summary.get(f"grad_norm/{b}/reward_std_mean", 0) for b in BUCKETS}
    for ax, comp, color in zip(axes, COMPONENTS, colors):
        y_values = []
        for bucket in BUCKETS:
            # Construct key based on AgentTrainer matching
            key = f"grad_norm/{bucket}/{comp}"
            val = summary.get(key, 0)
            y_values.append(val)
            
        # Draw bar plot
        bars = ax.bar(x_labels, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_title(titles[comp], fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Grad Norm Magnitude", fontsize=12)
        ax.set_xlabel("Reward Variance Bucket", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        legend_lines = []
        for label, bucket in zip(x_labels, BUCKETS):
            rv = bucket_rv.get(bucket, 0)
            legend_lines.append(f"{label}: rv={rv:.4f}")
        ax.legend(
            [plt.Line2D([0], [0], color="none")],
            ["  ".join(legend_lines)],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            fontsize=9,
        )
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(y_values)*0.01 if y_values else 0.01),
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle(f"Gradient Norms - Run: {run.name}", fontsize=20, y=1.05)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300)
    print(f"\nSuccess! Results visualization saved to: {os.path.abspath(OUTPUT_FILE)}")

    # Create subplots for per-component losses
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    loss_titles = {
        "policy": "Policy (Task) Loss",
        "entropy": "Entropy Loss",
        "kl": "KL Loss",
        "total": "Total Loss",
    }
    loss_colors = ["#8e44ad", "#27ae60", "#2980b9", "#c0392b"]  # Purple, Green, Blue, Red

    for ax, comp, color in zip(axes2.flatten(), LOSS_COMPONENTS, loss_colors):
        y_values = []
        for bucket in BUCKETS:
            key = f"grad_norm/{bucket}/loss/{comp}"
            val = summary.get(key, 0)
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

    fig2.suptitle(f"Per-Component Losses - Run: {run.name}", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_LOSS, bbox_inches="tight", dpi=300)
    print(f"Success! Loss visualization saved to: {os.path.abspath(OUTPUT_FILE_LOSS)}")

    # Create plot for per-bucket mean reward variance (std)
    rv_values = []
    for bucket in BUCKETS:
        rv_values.append(summary.get(f"grad_norm/{bucket}/reward_std_mean", 0))
    if any(v != 0 for v in rv_values):
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
        bars = ax3.bar(x_labels, rv_values, color="#f39c12", alpha=0.85, edgecolor="black", linewidth=1)
        ax3.set_title(f"Reward Std Mean by Bucket - Run: {run.name}", fontsize=14, fontweight="bold", pad=10)
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
        plt.savefig(OUTPUT_FILE_RV, bbox_inches="tight", dpi=300)
        print(f"Success! Reward std visualization saved to: {os.path.abspath(OUTPUT_FILE_RV)}")
    else:
        print("Warning: No reward std mean metrics found; skipping reward std plot.")

    # Create plots for per-sample and per-token grad norms (combined per component)
    fig4, axes4 = plt.subplots(1, 3, figsize=(20, 6))
    plt.subplots_adjust(wspace=0.3)
    norm_titles = {
        "kl": "KL Grad Norm (Per Sample vs Per Token)",
        "entropy": "Entropy Grad Norm (Per Sample vs Per Token)",
        "task": "Task Grad Norm (Per Sample vs Per Token)",
    }
    for ax, comp in zip(axes4, COMPONENTS):
        per_sample = []
        per_token = []
        for bucket in BUCKETS:
            per_sample.append(summary.get(f"grad_norm/{bucket}/per_sample/{comp}", 0))
            per_token.append(summary.get(f"grad_norm/{bucket}/per_token/{comp}", 0))

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

    fig4.suptitle(f"Normalized Grad Norms - Run: {run.name}", fontsize=18, y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_NORMED, bbox_inches="tight", dpi=300)
    print(f"Success! Normalized grad visualization saved to: {os.path.abspath(OUTPUT_FILE_NORMED)}")
    
    # Also print a summary table to console for quick reference
    print("\nSummary Table (Grad Norms):")
    print("-" * 70)
    header = f"{'Bucket':<15} | {'KL':<12} | {'Entropy':<12} | {'Task':<12}"
    print(header)
    print("-" * 70)
    for i, bucket in enumerate(BUCKETS):
        kl = summary.get(f"grad_norm/{bucket}/kl", 0)
        ent = summary.get(f"grad_norm/{bucket}/entropy", 0)
        tsk = summary.get(f"grad_norm/{bucket}/task", 0)
        print(f"{x_labels[i]:<15} | {kl:<12.5f} | {ent:<12.5f} | {tsk:<12.5f}")
    print("-" * 70)

    print("\nSummary Table (Losses):")
    print("-" * 86)
    header = f"{'Bucket':<15} | {'Policy':<12} | {'Entropy':<12} | {'KL':<12} | {'Total':<12}"
    print(header)
    print("-" * 86)
    for i, bucket in enumerate(BUCKETS):
        policy = summary.get(f"grad_norm/{bucket}/loss/policy", 0)
        ent = summary.get(f"grad_norm/{bucket}/loss/entropy", 0)
        kl = summary.get(f"grad_norm/{bucket}/loss/kl", 0)
        total = summary.get(f"grad_norm/{bucket}/loss/total", 0)
        print(f"{x_labels[i]:<15} | {policy:<12.5f} | {ent:<12.5f} | {kl:<12.5f} | {total:<12.5f}")
    print("-" * 86)

    print("\nSummary Table (Reward Std Mean):")
    print("-" * 50)
    header = f"{'Bucket':<15} | {'Reward Std Mean':<16}"
    print(header)
    print("-" * 50)
    for i, bucket in enumerate(BUCKETS):
        rv = summary.get(f"grad_norm/{bucket}/reward_std_mean", 0)
        print(f"{x_labels[i]:<15} | {rv:<16.5f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
