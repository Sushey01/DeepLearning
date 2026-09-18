"""
Step 3 of the pipeline: evaluates trained checkpoints on the held-out
test set and writes a macro P/R/F1 report, confusion matrices, and a
model-comparison summary.

Usage:
    python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from dataset import build_dataloaders
from model import build_model, count_params
from utils import get_device, load_config


def read_log_metadata(log_path: Path):
    """Return training_time_sec and peak_gpu_memory_mb if present in the log CSV."""
    if not log_path.exists():
        return {"training_time_sec": None, "peak_gpu_memory_mb": None}

    try:
        df = pd.read_csv(log_path)
    except Exception:
        return {"training_time_sec": None, "peak_gpu_memory_mb": None}

    info = {"training_time_sec": None, "peak_gpu_memory_mb": None}
    if {"training_time_sec", "peak_gpu_memory_mb"}.issubset(df.columns):
        last = df.iloc[-1]
        info["training_time_sec"] = float(last["training_time_sec"])
        info["peak_gpu_memory_mb"] = float(last["peak_gpu_memory_mb"])
    return info


def evaluate_model(cfg: dict, model_name: str, device: torch.device):
    classes = cfg["classes"]
    _, _, test_loader, _ = build_dataloaders(cfg)

    model = build_model(model_name, num_classes=len(classes)).to(device)
    checkpoint_path = Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_probs.append(probs.numpy())

    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((len(all_labels), len(classes)))
    report = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0
    )
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    truth = label_binarize(all_labels, classes=list(range(len(classes))))
    if truth.shape[0] > 0 and truth.shape[1] == len(classes):
        try:
            macro_auc = roc_auc_score(truth, all_probs, average="macro", multi_class="ovr")
        except ValueError:
            macro_auc = float("nan")
    else:
        macro_auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds, normalize="true")
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    report_payload = dict(report)
    report_payload["macro_f1"] = float(report["macro avg"]["f1-score"])
    report_payload["macro_auc"] = float(macro_auc) if not np.isnan(macro_auc) else None
    report_payload["accuracy"] = float(report["accuracy"])
    report_payload["params_m"] = count_params(model) / 1_000_000
    log_meta = read_log_metadata(Path(cfg["logs_dir"]) / f"{model_name}_log.csv")
    report_payload["training_time_sec"] = log_meta["training_time_sec"]
    report_payload["peak_gpu_memory_mb"] = log_meta["peak_gpu_memory_mb"]

    with open(results_dir / f"{model_name}_report.json", "w") as f:
        json.dump(report_payload, f, indent=2)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalised Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "model": model_name,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_auc": float(macro_auc) if not np.isnan(macro_auc) else None,
        "accuracy": float(report["accuracy"]),
        "params_m": count_params(model) / 1_000_000,
        "training_time_sec": log_meta["training_time_sec"],
        "peak_gpu_memory_mb": log_meta["peak_gpu_memory_mb"],
        "cm": cm,
    }


def generate_combined_confusion_matrix(cfg: dict, results_dir: Path, class_names: list, per_model: dict):
    fig, axes = plt.subplots(1, len(per_model), figsize=(5 * len(per_model), 4.5), squeeze=False)
    axes = axes[0]
    vmin, vmax = 0.0, 1.0

    for ax, (model_name, values) in zip(axes, per_model.items()):
        sns.heatmap(
            values["cm"],
            annot=True,
            fmt=".2f",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=(ax is axes[-1]),
        )
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    if len(per_model) > 0:
        fig.colorbar(axes[-1].collections[0], ax=axes, shrink=0.9)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrices_all.png", dpi=200)
    plt.close(fig)


def build_comparison_table(per_model: dict):
    rows = []
    for model_name, values in per_model.items():
        rows.append({
            "Model": model_name,
            "Macro-F1": values["macro_f1"],
            "Macro-AUC": values["macro_auc"],
            "Accuracy": values["accuracy"],
            "Params(M)": values["params_m"],
            "Training Time": values["training_time_sec"],
            "Peak GPU Memory": values["peak_gpu_memory_mb"],
        })
    return rows


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    device = get_device()
    classes = cfg["classes"]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    requested = [model_name]
    if model_name == "all":
        requested = list(cfg.get("models", ["efficientnet_b0", "mobilenet_v3_small", "densenet121"]))

    per_model = {}
    for name in requested:
        metrics = evaluate_model(cfg, name, device)
        per_model[name] = metrics

    comparison_rows = build_comparison_table(per_model)
    with open(results_dir / "comparison_table.json", "w") as f:
        json.dump(comparison_rows, f, indent=2)

    print("\nModel comparison summary")
    print(f"{'Model':<22} {'Macro-F1':>9} {'Macro-AUC':>10} {'Accuracy':>9} {'Params(M)':>10} {'Training Time':>13} {'Peak GPU Memory':>16}")
    for row in comparison_rows:
        print(
            f"{row['Model']:<22} "
            f"{row['Macro-F1'] if row['Macro-F1'] is not None else 'N/A':>9} "
            f"{row['Macro-AUC'] if row['Macro-AUC'] is not None else 'N/A':>10} "
            f"{row['Accuracy'] if row['Accuracy'] is not None else 'N/A':>9} "
            f"{row['Params(M)'] if row['Params(M)'] is not None else 'N/A':>10} "
            f"{row['Training Time'] if row['Training Time'] is not None else 'N/A':>13} "
            f"{row['Peak GPU Memory'] if row['Peak GPU Memory'] is not None else 'N/A':>16}"
        )

    if len(per_model) > 1:
        generate_combined_confusion_matrix(cfg, results_dir, classes, per_model)
        print(f"\nSaved combined confusion matrices to {results_dir / 'confusion_matrices_all.png'}")

    print(f"\nSaved comparison table to {results_dir / 'comparison_table.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121", "all"],
    )
    args = parser.parse_args()
    main(args.config, args.model)
