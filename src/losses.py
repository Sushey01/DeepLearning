"""
Loss functions for class imbalance.

compute_class_weights(): uses the "effective number of samples" formula
(Cui et al., 2019) rather than raw inverse frequency, which is gentler and
more stable when one class (Platelet, ~543 crops) is extremely small.

FocalLoss: alternative to weighted CE — down-weights easy/majority-class
examples so gradient stays focused on hard/rare ones. Try both and compare
validation macro-F1 (see README).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(class_counts: list, beta: float = None) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float)

    if beta is None:
        # raw inverse frequency fallback
        weights = 1.0 / counts
    else:
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num

    weights = weights / weights.sum() * len(counts)  # normalise so avg weight ~1
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_loss(cfg: dict, class_counts: list, device) -> nn.Module:
    weights = compute_class_weights(class_counts, cfg.get("effective_number_beta")).to(device)

    if cfg["loss_type"] == "focal":
        return FocalLoss(alpha=weights, gamma=cfg.get("focal_gamma", 2.0))
    return nn.CrossEntropyLoss(weight=weights)
