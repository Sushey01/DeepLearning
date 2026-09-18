"""Generate Grad-CAM heatmaps for trained blood-cell models.

This script follows the repo's existing evaluation conventions and is kept as an
optional enhancement, without altering the training pipeline.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.utils import get_device, load_config


def get_target_layer(model, model_name: str):
    """Return the final convolutional block for each backbone."""
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "mobilenet_v3_small":
        return model.features[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap


def overlay_heatmap(image_np, heatmap_np):
    heatmap = np.array(Image.fromarray(heatmap_np).resize((image_np.shape[1], image_np.shape[0])))
    heatmap = normalize_heatmap(heatmap)
    jet = plt.cm.get_cmap("jet")
    heatmap_color = (jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.6 * image_np + 0.4 * heatmap_color).astype(np.uint8)
    return overlay


def create_gradcam(model, image_tensor, target_class_idx, model_name: str):
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    target_layer = get_target_layer(model, model_name)
    activations = []
    gradients = []

    def _forward_hook(module, inputs, output):
        activations.append(output.detach())

    def _backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    handle_f = target_layer.register_forward_hook(_forward_hook)
    handle_b = target_layer.register_full_backward_hook(_backward_hook)

    try:
        logits = model(image_tensor)
        model.zero_grad()
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)
    finally:
        handle_f.remove()
        handle_b.remove()

    if not activations or not gradients:
        raise RuntimeError(f"Failed to capture activations/gradients for {model_name}")

    activation_map = activations[-1]
    grad_map = gradients[-1]

    if activation_map.dim() == 4 and activation_map.shape[0] == 1:
        activation_map = activation_map[0]
    if grad_map.dim() == 4 and grad_map.shape[0] == 1:
        grad_map = grad_map[0]

    weights = grad_map.mean(dim=(1, 2))
    cam = (weights[:, :, None, None] * activation_map).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.cpu().numpy()[0]
    return normalize_heatmap(cam)


def load_image_for_gradcam(image_path: str):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image)
    return tensor.unsqueeze(0), np.array(image)


def run_gradcam(config_path: str, model_name: str, checkpoint: str = None):
    cfg = load_config(config_path)
    classes = cfg["classes"]
    device = get_device()
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, num_classes=len(classes)).to(device)
    checkpoint_path = Path(checkpoint) if checkpoint else Path(cfg["checkpoints_dir"]) / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    sample_dir = Path(cfg["processed_dir"]) / "test"
    class_sample_paths = []
    for cls_name in classes:
        cls_dir = sample_dir / cls_name
        if cls_dir.exists():
            files = sorted(cls_dir.glob("*.png"))
            if files:
                class_sample_paths.append((cls_name, files[0]))

    if not class_sample_paths:
        raise FileNotFoundError(f"No processed test samples found under {sample_dir}")

    for cls_name, image_path in class_sample_paths:
        input_tensor, original_img = load_image_for_gradcam(str(image_path))
        input_tensor = input_tensor.to(device)
        target_class = classes.index(cls_name)
        cam = create_gradcam(model, input_tensor, target_class, model_name)
        output = overlay_heatmap(original_img, cam)
        out_path = results_dir / f"gradcam_{cls_name}_correct.png"
        Image.fromarray(output).save(out_path)
        print(f"Saved Grad-CAM overlay for {cls_name}: {out_path}")

    print(f"Grad-CAM run for model {model_name} complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["efficientnet_b0", "mobilenet_v3_small", "densenet121"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    run_gradcam(args.config, args.model, args.checkpoint)
