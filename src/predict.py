import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model import build_model
from utils import get_device, load_config


def load_model(model_name: str, config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    classes = cfg["classes"]
    device = get_device()

    model = build_model(model_name, num_classes=len(classes)).to(device)
    checkpoint_path = Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, classes, cfg, device


def preprocess_image(image_path: str, cfg: dict):
    image = Image.open(image_path).convert("RGB")
    return preprocess_image_object(image, cfg)


def preprocess_image_object(image: Image.Image, cfg: dict):
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["imagenet_mean"], std=cfg["imagenet_std"]),
    ])
    return transform(image)


def preprocess_image_bytes(image_bytes: bytes, cfg: dict):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return preprocess_image_object(image, cfg)


def predict_image(image_path: str, model, classes: list, cfg: dict, device, threshold: float = 0.0):
    x = preprocess_image(image_path, cfg).unsqueeze(0).to(device)
    return predict_tensor(model, x, classes, cfg, device, image_path, threshold)


def predict_image_bytes(image_bytes: bytes, model, classes: list, cfg: dict, device, threshold: float = 0.0, image_name: str = "uploaded_image"):
    x = preprocess_image_bytes(image_bytes, cfg).unsqueeze(0).to(device)
    return predict_tensor(model, x, classes, cfg, device, image_name, threshold)


def predict_tensor(model, x, classes: list, cfg: dict, device, image_name: str, threshold: float = 0.0):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_label = classes[pred_idx]
        confidence = float(probs[pred_idx].item())

    result = {
        "image": image_name,
        "predicted_class": pred_label,
        "confidence": confidence,
        "probabilities": {cls: float(probs[i].item()) for i, cls in enumerate(classes)},
    }

    if threshold > 0 and confidence < threshold:
        result["warning"] = f"Low confidence prediction below threshold {threshold:.2f}"

    return result


def predict_folder(folder_path: str, model_name: str, config_path: str = "configs/config.yaml", output_csv: str = None, limit: int = None, threshold: float = 0.0):
    model, classes, cfg, device = load_model(model_name, config_path)
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if limit is not None:
        image_paths = image_paths[:limit]

    if not image_paths:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    rows = []
    for image_path in image_paths:
        result = predict_image(str(image_path), model, classes, cfg, device, threshold=threshold)
        rows.append({
            "image": image_path.name,
            "predicted_class": result["predicted_class"],
            "confidence": round(result["confidence"], 4),
            "warning": result.get("warning", "")
        })
        print(f"{image_path.name}: {result['predicted_class']} ({result['confidence']:.4f})")
        if "warning" in result:
            print(f"  Warning: {result['warning']}")

    if output_csv:
        output_file = Path(output_csv)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "predicted_class", "confidence", "warning"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved batch predictions to {output_file}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict blood cell class for one image or a folder of images.")
    parser.add_argument("--image", type=str, help="Path to a single blood cell image.")
    parser.add_argument("--folder", type=str, help="Path to a folder containing blood cell images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121"], help="Model to use for prediction.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV file to save predictions for a folder.")
    parser.add_argument("--limit", type=int, default=None, help="Limit batch prediction to the first N images.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Optional confidence threshold; low confidence predictions are flagged.")
    args = parser.parse_args()

    if args.image and args.folder:
        raise ValueError("Provide either --image or --folder, not both.")
    if not args.image and not args.folder:
        raise ValueError("Please provide either --image or --folder.")

    model, classes, cfg, device = load_model(args.model, args.config)

    if args.image:
        result = predict_image(args.image, model, classes, cfg, device, threshold=args.threshold)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        for cls, prob in result["probabilities"].items():
            print(f"  {cls}: {prob:.4f}")
        if "warning" in result:
            print(f"Warning: {result['warning']}")
    else:
        predict_folder(args.folder, args.model, args.config, output_csv=args.output_csv, limit=args.limit, threshold=args.threshold)
