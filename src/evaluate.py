"""
Step 3 of the pipeline: evaluates a trained checkpoint on the held-out
test set and writes a macro P/R/F1 report + normalised confusion matrix.

Usage:
    python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders
from model import build_model
from utils import get_device, load_config


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    device = get_device()
    classes = cfg["classes"]

    _, _, test_loader, _ = build_dataloaders(cfg)

    model = build_model(model_name, num_classes=len(classes)).to(device)
    checkpoint_path = Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    report = classification_report(
        all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0
    )
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f"{model_name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalised Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"{model_name}_confusion_matrix.png", dpi=150)

    print(f"\nSaved report + confusion matrix to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, required=True,
                         choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121"])
    args = parser.parse_args()
    main(args.config, args.model)
