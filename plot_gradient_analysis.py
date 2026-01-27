import wandb
import matplotlib.pyplot as plt
import os

# CONFIGURATION
WANDB_PATH = "deimos-xing/AGEN_gradient_analysis/8ft408fq"
BUCKETS = [
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
]
COMPONENTS = ["kl", "entropy", "task"]
LOSS_COMPONENTS = ["policy", "entropy", "kl", "total"]
OUTPUT_FILE = "gradient_analysis_plots.png"
OUTPUT_FILE_LOSS = "gradient_analysis_loss_plots.png"

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

if __name__ == "__main__":
    main()
