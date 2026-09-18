# Blood Cell Classification (CMP6232)

This project trains a transfer-learning classifier for RBC / WBC / Platelet cell crops using the TXL-PBC dataset, comparing three backbones:

- EfficientNet-B0
- MobileNetV3-Small
- DenseNet121

The current codebase includes the original training pipeline, evaluation scripts, and the requested enhancement scripts for reporting and explainability support.

## Project structure

```
blood-cell-classification/
├── data/
│   ├── raw/
│   │   ├── images/         # source microscopy images
│   │   ├── labels/         # YOLO-format annotation files
│   │   ├── train.txt       # optional TXL-PBC split list
│   │   ├── val.txt         # optional TXL-PBC split list
│   │   └── test.txt        # optional TXL-PBC split list
│   └── processed/
│       ├── train/{RBC,WBC,Platelet}/
│       ├── val/{RBC,WBC,Platelet}/
│       └── test/{RBC,WBC,Platelet}/
├── configs/
│   └── config.yaml         # dataset paths, class list, model names, and hyperparameters
├── src/
│   ├── crop_dataset.py     # raw images + YOLO boxes -> cropped per-class dataset
│   ├── dataset.py          # PyTorch Dataset, transforms, and weighted sampler
│   ├── model.py            # EfficientNet-B0 / MobileNetV3-Small / DenseNet121 builders
│   ├── losses.py           # class-weighted CE and focal loss
│   ├── train.py            # training loop, checkpointing on best validation macro-F1
│   ├── evaluate.py         # evaluation, confusion matrices, comparison output, runtime/memory metadata
│   ├── gradcam.py          # optional Grad-CAM heatmap overlay script
│   ├── plot_model_results.py  # loss-curve plotting script
│   ├── reporting.py        # summary text generation utilities
│   └── utils.py            # config loading and shared helpers
├── outputs/
│   ├── checkpoints/       # saved model weights (best_<model_name>.pt)
│   ├── logs/               # per-epoch CSV logs with loss/F1 and timing metadata
│   └── results/            # confusion matrices, JSON reports, comparison figures, Grad-CAM outputs
├── tests/
│   ├── test_api_inference.py
│   ├── test_enhancement_features.py
│   └── test_requested_enhancements.py
├── plan/
│   ├── project-enhancement-plan.md
│   └── file-role-summary.md
├── requirements.txt
├── README.md
└── finalreport.tex
```

## Data and class convention

The project assumes the TXL-PBC YOLO annotation order is:

- 0 = WBC
- 1 = RBC
- 2 = Platelet

This is reflected in [configs/config.yaml](configs/config.yaml) and enforced in [src/crop_dataset.py](src/crop_dataset.py). The model name for MobileNetV3-Small is kept as `mobilenet_v3_small` throughout the project.

## Setup and environment

This project is intended for Python 3.11. On Windows, Python 3.14 can resolve to the Microsoft Store alias, which can lead to environment and CUDA setup issues. Use the interpreter explicitly if needed.

```powershell
& "C:\Users\MSI\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check CUDA availability before training:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

If you need a CUDA-enabled PyTorch build on Windows with NVIDIA GPUs:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Workflow

1. Download the TXL-PBC dataset and place the source microscopy images in `data/raw/images/` and the YOLO label files in `data/raw/labels/`.

2. Crop the dataset from raw images and annotations:

```bash
python src/crop_dataset.py --config configs/config.yaml
```

3. Sanity-check the class counts after cropping. The expected class order and folder layout should match the source YOLO labels; if not, the labels/configuration should be reviewed before training.

4. Train a single model at a time:

```bash
python src/train.py --config configs/config.yaml --model efficientnet_b0
python src/train.py --config configs/config.yaml --model mobilenet_v3_small
python src/train.py --config configs/config.yaml --model densenet121
```

Each run writes per-epoch logs to `outputs/logs/`, saves the best checkpoint to `outputs/checkpoints/`, and records training time plus peak GPU memory in the CSV log.

5. Evaluate a model on the test set:

```bash
python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
```

This writes a normalised confusion matrix, JSON metrics, and a model comparison payload to `outputs/results/`.

6. Generate the combined comparison visuals:

```bash
python src/evaluate.py --config configs/config.yaml --model all
python src/plot_model_results.py --config configs/config.yaml
```

These commands produce:

- `outputs/results/confusion_matrices_all.png`
- `outputs/results/loss_curves.png`
- `outputs/results/comparison_table.json`

7. Run Grad-CAM on a trained checkpoint if explainability is desired:

```bash
python src/gradcam.py --config configs/config.yaml --model efficientnet_b0
```

This is an optional enhancement and is not part of the core training/evaluation pipeline.

## Notes

- Class imbalance handling is already built into the dataset and loss setup through weighted sampling and class-aware loss weighting.
- Checkpoint selection is driven by validation macro-F1, not raw accuracy.
- The project is implemented as a PyTorch transfer-learning pipeline rather than a Keras/TensorFlow pipeline.
- The added evaluation and reporting scripts are meant to improve final reporting and comparison quality without changing the model-training behavior itself.

## Model comparison summary

The current project output is designed to compare the three backbone models on the same held-out test set using:

- Macro-F1
- Macro-AUC
- Accuracy
- Parameter count
- Training time
- Peak GPU memory

The final report should use the model with the strongest macro-F1 value as the primary recommendation, while using accuracy and efficiency as secondary checks.
