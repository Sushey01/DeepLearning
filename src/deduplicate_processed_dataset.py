#!/usr/bin/env python3
"""Audit near-duplicate cell crops across processed splits without deleting anything until approved."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from imagehash import phash

CLASSES = ["WBC", "RBC", "Platelet"]
SPLITS = ["train", "val", "test"]
SPLIT_ORDER = {split: i for i, split in enumerate(SPLITS)}
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("outputs/results")

_PHASH_CACHE = {}


def list_processed_images():
    records = []
    for split in SPLITS:
        for cls in CLASSES:
            split_dir = PROCESSED_DIR / split / cls
            if not split_dir.exists():
                continue
            for img_path in sorted(split_dir.glob("*.png")):
                records.append({"path": img_path, "split": split, "class": cls})
    return records


def compute_phash(path: Path):
    key = str(path)
    if key not in _PHASH_CACHE:
        _PHASH_CACHE[key] = phash(Image.open(path).convert("RGB"))
    return _PHASH_CACHE[key]


def split_pair_key(a_split: str, b_split: str) -> str:
    if a_split == b_split:
        raise ValueError("Split pair must be different")
    first, second = sorted([a_split, b_split], key=lambda split: SPLIT_ORDER[split])
    return f"{first}_{second}"


def pairwise_counts(records):
    pair_results = {
        "train_val": {cls: {"count": 0, "examples": []} for cls in CLASSES},
        "train_test": {cls: {"count": 0, "examples": []} for cls in CLASSES},
        "val_test": {cls: {"count": 0, "examples": []} for cls in CLASSES},
    }
    split_records = {split: {cls: [] for cls in CLASSES} for split in SPLITS}
    for rec in records:
        split_records[rec["split"]][rec["class"]].append(rec)

    for split_a in SPLITS:
        for split_b in SPLITS:
            if SPLIT_ORDER[split_a] >= SPLIT_ORDER[split_b]:
                continue
            pair_key = split_pair_key(split_a, split_b)
            for cls in CLASSES:
                examples = []
                count = 0
                for rec_a in split_records[split_a][cls]:
                    h_a = compute_phash(rec_a["path"])
                    for rec_b in split_records[split_b][cls]:
                        dist = h_a - compute_phash(rec_b["path"])
                        if dist <= 5:
                            count += 1
                            if len(examples) < 5:
                                examples.append({"a": rec_a, "b": rec_b, "distance": int(dist)})
                pair_results[pair_key][cls] = {"count": count, "examples": examples}
    return pair_results


def save_pair_examples(pair_results):
    pair_dir = RESULTS_DIR / "pairwise_phash"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for key in ["train_val", "train_test", "val_test"]:
        sample_rows = []
        for cls in CLASSES:
            sample_rows.extend(pair_results[key][cls]["examples"])
        if not sample_rows:
            continue
        sample_rows = sample_rows[:5]
        montage_h = 224 * len(sample_rows)
        montage = Image.new("RGB", (448, montage_h), color=(255, 255, 255))
        for idx, rec in enumerate(sample_rows):
            img_a = Image.open(rec["a"]["path"]).convert("RGB").resize((224, 224))
            img_b = Image.open(rec["b"]["path"]).convert("RGB").resize((224, 224))
            montage.paste(img_a, (0, idx * 224))
            montage.paste(img_b, (224, idx * 224))
        montage.save(pair_dir / f"{key}_examples.png")


def split_class_counts(records):
    counts = {split: {cls: 0 for cls in CLASSES} for split in SPLITS}
    for rec in records:
        counts[rec["split"]][rec["class"]] += 1
    return counts


def build_keep_map(records):
    keep_map = {}
    for cls in CLASSES:
        class_records = sorted(
            [rec for rec in records if rec["class"] == cls],
            key=lambda rec: (SPLIT_ORDER[rec["split"]], str(rec["path"])),
        )
        kept = []
        for rec in class_records:
            rec_hash = compute_phash(rec["path"])
            if any(rec_hash - compute_phash(existing["path"]) <= 5 for existing in kept):
                continue
            kept.append(rec)
        for rec in kept:
            keep_map[str(rec["path"])] = rec
    return keep_map


def deduplicate_processed_dataset(apply_cleanup: bool = False):
    records = list_processed_images()
    before_counts = split_class_counts(records)
    pair_results = pairwise_counts(records)
    save_pair_examples(pair_results)

    keep_map = build_keep_map(records)
    removed_paths = sorted({str(rec["path"]) for rec in records if str(rec["path"]) not in keep_map})
    removed_by_class = {cls: 0 for cls in CLASSES}
    for rec in records:
        if str(rec["path"]) in removed_paths:
            removed_by_class[rec["class"]] += 1

    if apply_cleanup:
        for path in removed_paths:
            p = Path(path)
            if p.exists():
                p.unlink()
        after_records = list_processed_images()
    else:
        after_records = records

    after_counts = split_class_counts(after_records)
    summary = {
        "before": before_counts,
        "after": after_counts,
        "removed_count": len(removed_paths),
        "removed_by_class": removed_by_class,
        "pairwise_counts": {
            key: {cls: pair_results[key][cls]["count"] for cls in CLASSES}
            for key in ["train_val", "train_test", "val_test"]
        },
        "dry_run": not apply_cleanup,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "deduplication_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["# Deduplication summary", ""]
    lines.append("## Before / after class counts")
    for split in SPLITS:
        lines.append(f"### {split}")
        lines.append("| Class | Before | After | Removed |")
        lines.append("| --- | ---: | ---: | ---: |")
        for cls in CLASSES:
            before = before_counts[split][cls]
            after = after_counts[split][cls]
            lines.append(f"| {cls} | {before} | {after} | {before - after} |")
        lines.append("")
    lines.append(f"- Images flagged for removal: {len(removed_paths)}")
    lines.append(f"- Dry run: {'yes' if not apply_cleanup else 'no'}")
    lines.append("")
    lines.append("## Pairwise pHash counts")
    for key in ["train_val", "train_test", "val_test"]:
        counts = {cls: pair_results[key][cls]["count"] for cls in CLASSES}
        total = sum(counts.values())
        lines.append(f"- {key}: {total} near-duplicate pairs")
        lines.append("  - by class: " + ", ".join(f"{cls}={counts.get(cls, 0)}" for cls in CLASSES))
    with open(RESULTS_DIR / "deduplication_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    apply_cleanup = "--apply" in sys.argv
    deduplicate_processed_dataset(apply_cleanup=apply_cleanup)
