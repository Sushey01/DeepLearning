"""Utilities for report generation and model comparison summaries."""

from __future__ import annotations


def build_model_summary(metrics: dict) -> str:
    """Build a markdown summary table for model comparison results.

    Example metrics structure:
    {
        "efficientnet_b0": {
            "accuracy": 0.99,
            "macro_f1": 0.985,
            "params": 5000000,
            "train_time_min": 8.2,
        }
    }
    """
    if not metrics:
        return "## Model comparison\n\nNo metrics available."

    best_model = max(
        metrics,
        key=lambda name: (
            float(metrics[name].get("macro_f1", 0.0)),
            float(metrics[name].get("accuracy", 0.0)),
        ),
    )
    best = metrics[best_model]

    lines = [
        "# Model comparison summary",
        "",
        "## Best model",
        f"The recommended model is **{best_model}** with macro F1 = {float(best.get('macro_f1', 0.0)):.4f} and accuracy = {float(best.get('accuracy', 0.0)):.4f}.",
        "",
        "| Model | Accuracy | Macro F1 | Params | Train Time (min) |",
        "|---|---:|---:|---:|---:|",
    ]

    for model_name, values in metrics.items():
        accuracy = float(values.get("accuracy", 0.0))
        macro_f1 = float(values.get("macro_f1", 0.0))
        params = int(values.get("params", 0))
        train_time = float(values.get("train_time_min", 0.0))
        lines.append(
            f"| {model_name} | {accuracy:.4f} | {macro_f1:.4f} | {params:,} | {train_time:.1f} |"
        )

    lines.extend([
        "",
        "## Recommendation",
        "Use the model with the highest macro F1 as the final reporting choice, with accuracy and training efficiency as secondary checks.",
    ])
    return "\n".join(lines)
