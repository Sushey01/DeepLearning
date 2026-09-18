"""Generate aggregated training/evaluation plots for the project.

This script reads the per-model CSV logs from outputs/logs/ and creates a
combined loss-curve plot.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_config


def plot_loss_curves(config_path: str):
    cfg = load_config(config_path)
    logs_dir = Path(cfg["logs_dir"])
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["efficientnet_b0", "mobilenet_v3_small", "densenet121"]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for model_name in model_names:
        csv_path = logs_dir / f"{model_name}_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "train_loss" in df.columns and "val_loss" in df.columns:
            ax.plot(df.index + 1, df["train_loss"], label=f"{model_name} train loss", alpha=0.8)
            ax.plot(df.index + 1, df["val_loss"], label=f"{model_name} val loss", linestyle="--", alpha=0.9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curves by Model")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "loss_curves.png", dpi=200)
    plt.close(fig)
    print(f"Saved loss-curve plot to {results_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    plot_loss_curves(args.config)
