"""
Step 2 of the pipeline: trains one model at a time.

Usage:
    python src/train.py --config configs/config.yaml --model efficientnet_b0
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import build_dataloaders
from losses import build_loss
from model import build_model, count_params
from utils import get_device, load_config, set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def main(config_path: str, model_name: str):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    train_loader, val_loader, _, class_counts = build_dataloaders(cfg)
    print(f"Training class counts: {dict(zip(cfg['classes'], class_counts))}")

    model = build_model(model_name, num_classes=len(cfg["classes"])).to(device)
    print(f"{model_name}: {count_params(model):,} trainable params")

    criterion = build_loss(cfg, class_counts, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    Path(cfg["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logs_dir"]).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["logs_dir"]) / f"{model_name}_log.csv"
    checkpoint_path = Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    start_time = time.time()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_f1", "val_loss", "val_f1"])

        for epoch in tqdm(range(1, cfg["num_epochs"] + 1), desc=f"Training {model_name}"):
            train_loss, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            val_loss, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

            writer.writerow([epoch, train_loss, train_f1, val_loss, val_f1])
            f.flush()
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch} (best val macro-F1={best_val_f1:.4f})")
                    break

    elapsed = time.time() - start_time
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )
    print(f"\n{model_name} done. Best val macro-F1={best_val_f1:.4f}. "
          f"Training time={elapsed/60:.1f} min. Peak GPU memory={peak_mem_mb:.0f} MB.")
    print(f"Best checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, required=True,
                         choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121"])
    args = parser.parse_args()
    main(args.config, args.model)
