import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import build_model
from utils import get_device, load_config


def predict_single_image(image_path: str, config_path: str = "configs/config.yaml", model_name: str = "efficientnet_b0"):
    cfg = load_config(config_path)
    classes = cfg["classes"]
    device = get_device()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((cfg["image_size"], cfg["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["imagenet_mean"], std=cfg["imagenet_std"]),
    ])

    x = transform(image).unsqueeze(0).to(device)

    model = build_model(model_name, num_classes=len(classes)).to(device)
    checkpoint_path = Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(x)
        pred_idx = logits.argmax(dim=1).item()

    pred_label = classes[pred_idx]
    probs = torch.softmax(logits, dim=1)[0]
    prob = probs[pred_idx].item()

    print(f"Predicted class: {pred_label}")
    print(f"Confidence: {prob:.4f}")
    for idx, cls in enumerate(classes):
        print(f"  {cls}: {probs[idx].item():.4f}")

    return pred_label, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict class of a single blood cell image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input blood cell image.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121"], help="Model to use for prediction.")
    args = parser.parse_args()

    predict_single_image(args.image, args.config, args.model)
