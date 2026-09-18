"""
Step 1 of the pipeline.

Reads YOLO-format labels (class x_center y_center width height, all
normalised 0-1) alongside their source images, crops each bounding box out,
and saves it into data/processed/<split>/<class>/<uid>.png.

Split assignment is done at the IMAGE level (using the provided split
files if available, otherwise a random stratified fallback), so multiple
crops from the same source image never end up split across train/val/test.

Usage:
    python src/crop_dataset.py --config configs/config.yaml
"""
import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from utils import load_config, set_seed


def read_split_file(path: str) -> set:
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, "r") as f:
        return {Path(line.strip()).stem for line in f if line.strip()}


def assign_splits(image_stems: list, cfg: dict) -> dict:
    """Return {stem: split_name}.

    The current TXL-PBC project uses explicit split files on disk, which resolve to
    approximately 70/20/10 for train/val/test. If those files are absent, the
    fallback is a random 70/15/15 image-level split.
    """
    train_set = read_split_file(cfg["split_files"]["train"])
    val_set = read_split_file(cfg["split_files"]["val"])
    test_set = read_split_file(cfg["split_files"]["test"])

    if train_set or val_set or test_set:
        assignment = {}
        for stem in image_stems:
            if stem in train_set:
                assignment[stem] = "train"
            elif stem in val_set:
                assignment[stem] = "val"
            elif stem in test_set:
                assignment[stem] = "test"
            else:
                # image not listed in any split file — default to train
                assignment[stem] = "train"
        return assignment

    # Fallback: random split, done here at image level (not crop level)
    print("No split files found — falling back to a random 70/15/15 image-level split.")
    shuffled = image_stems.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    assignment = {}
    for i, stem in enumerate(shuffled):
        if i < n_train:
            assignment[stem] = "train"
        elif i < n_train + n_val:
            assignment[stem] = "val"
        else:
            assignment[stem] = "test"
    return assignment


def yolo_box_to_pixels(x_c, y_c, w, h, img_w, img_h):
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    images_dir = Path(cfg["raw_images_dir"])
    labels_dir = Path(cfg["raw_labels_dir"])
    processed_dir = Path(cfg["processed_dir"])
    # Keep this list aligned with the source YOLO label order.
    # TXL-PBC uses: 0=WBC, 1=RBC, 2=Platelet.
    classes = cfg["classes"]
    img_size = cfg["image_size"]

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(
            f"No .txt label files found in {labels_dir}. "
            "Point raw_labels_dir at your downloaded TXL-PBC labels."
        )

    image_stems = [lf.stem for lf in label_files]
    split_assignment = assign_splits(image_stems, cfg)

    # Make output dirs
    for split in ["train", "val", "test"]:
        for cls in classes:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counts = {split: {cls: 0 for cls in classes} for split in ["train", "val", "test"]}
    skipped = 0

    for label_file in tqdm(label_files, desc="Cropping cells"):
        stem = label_file.stem
        split = split_assignment.get(stem, "train")

        # Find the matching image (try common extensions)
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        with open(label_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_idx, x_c, y_c, w, h = parts
            cls_idx = int(cls_idx)
            if cls_idx >= len(classes):
                continue
            cls_name = classes[cls_idx]

            x1, y1, x2, y2 = yolo_box_to_pixels(
                float(x_c), float(y_c), float(w), float(h), img_w, img_h
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2)).resize((img_size, img_size))
            out_path = processed_dir / split / cls_name / f"{stem}_{i}.png"
            crop.save(out_path)
            counts[split][cls_name] += 1

    print("\nDone. Crop counts per split/class:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {counts[split]}")
    if skipped:
        print(f"\nWarning: skipped {skipped} label files with no matching image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
