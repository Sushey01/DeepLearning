# Blood Cell Classification (CMP6232)

This project trains a transfer-learning classifier for RBC / WBC / Platelet cell crops using the TXL-PBC dataset, comparing three backbones:

- EfficientNet-B0
- MobileNetV3-Small
- DenseNet121

The current codebase includes the original training pipeline, evaluation scripts, the duplicate-audit workflow, and the cleaned-data retraining outputs produced after deduplication.

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
│   ├── config.yaml                # baseline dataset paths and training config
│   └── config_post_dedup.yaml     # cleaned-dataset configuration used after deduplication
├── src/
│   ├── crop_dataset.py             # raw images + YOLO boxes -> cropped per-class dataset
│   ├── dataset.py                  # PyTorch Dataset, transforms, and weighted sampler
│   ├── model.py                    # EfficientNet-B0 / MobileNetV3-Small / DenseNet121 builders
│   ├── losses.py                   # class-weighted CE and focal loss
│   ├── train.py                    # training loop, checkpointing on best validation macro-F1
│   ├── evaluate.py                 # evaluation, confusion matrices, comparison output, runtime/memory metadata
│   ├── deduplicate_processed_dataset.py # dry-run and optional cleanup of near-duplicate crops
│   ├── gradcam.py                 # optional Grad-CAM heatmap overlay script
│   ├── plot_model_results.py      # loss-curve plotting script
│   ├── reporting.py                # summary text generation utilities
│   └── utils.py                   # config loading and shared helpers
├── outputs/
│   ├── checkpoints/               # saved baseline model weights (best_<model_name>.pt)
│   ├── logs/                      # baseline per-epoch CSV logs
│   ├── results/                   # baseline confusion matrices and JSON reports
│   └── report_images/             # report-ready baseline visuals
├── outputs_pre_dedup/
│   └── ...                        # preserved pre-cleanup baseline outputs
├── outputs_post_dedup/
│   ├── checkpoints/               # retrained model weights on the cleaned dataset
│   ├── logs/                      # cleaned-dataset training logs
│   ├── results/                   # cleaned-dataset evaluation reports and confusion matrices
│   └── post_dedup_results.md      # summary of the cleaned-data retraining run
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

### Split assignment

The project uses the TXL-PBC split files in [data/raw/train.txt](data/raw/train.txt), [data/raw/val.txt](data/raw/val.txt), and [data/raw/test.txt](data/raw/test.txt) when they are present. In the current dataset these resolve to approximately 70/20/10 for train/val/test (882 / 252 / 126 image IDs), which is the active runtime split used by the pipeline.

If the split files are missing, the fallback logic in [src/crop_dataset.py](src/crop_dataset.py) creates a random 70/15/15 image-level split. This fallback is only used when the explicit TXL-PBC split files are absent; it is not the current active setting for this project.

### Near-duplicate audit and cleanup

The project also includes a duplicate-image audit for processed crops. The script [src/deduplicate_processed_dataset.py](src/deduplicate_processed_dataset.py) checks perceptual-hash similarity for within-class images across train/val/test, reports pairwise counts, and saves montage examples for review.

Usage:

```bash
python src/deduplicate_processed_dataset.py
python src/deduplicate_processed_dataset.py --apply
```

- The default mode is a dry run and writes summary evidence to `outputs/results/`.
- The `--apply` flag removes near-duplicate crops from the processed dataset after review.
- The duplicate audit identified 694 near-duplicate images and reduced the cleaned test set from 1,881 crops to 1,768 crops.
- The project preserves both baseline and cleaned-data artifacts:
  - `outputs_pre_dedup/` = original pre-cleanup results
  - `outputs_post_dedup/` = retraining and evaluation performed on the deduplicated dataset

The cleaned dataset counts are:

| Split | WBC | RBC | Platelet | Total |
|---|---:|---:|---:|---:|
| Train | 901 | 10,854 | 382 | 12,137 |
| Validation | 252 | 3,180 | 112 | 3,544 |
| Test | 127 | 1,593 | 48 | 1,768 |

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

4. Audit and optionally clean near-duplicate images:

```bash
python src/deduplicate_processed_dataset.py
python src/deduplicate_processed_dataset.py --apply
```

This removes 694 near-duplicate crops from the processed dataset, producing the cleaned splits described above. The cleaned-data configuration is available at `configs/config_post_dedup.yaml`.

5. Train a single model on the chosen dataset version:

```bash
python src/train.py --config configs/config.yaml --model efficientnet_b0
python src/train.py --config configs/config.yaml --model mobilenet_v3_small
python src/train.py --config configs/config.yaml --model densenet121
```

For the cleaned dataset and the preserved post-dedup training artifacts, use:

```bash
python src/train.py --config configs/config_post_dedup.yaml --model efficientnet_b0
python src/train.py --config configs/config_post_dedup.yaml --model mobilenet_v3_small
python src/train.py --config configs/config_post_dedup.yaml --model densenet121
```

Each run writes per-epoch logs to the configured `logs_dir`, saves the best checkpoint to the configured `checkpoints_dir`, and records training time plus peak GPU memory in the CSV log.

6. Evaluate a model on the test set:

```bash
python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
python src/evaluate.py --config configs/config_post_dedup.yaml --model efficientnet_b0
```

This writes a normalised confusion matrix, JSON metrics, and a model comparison payload to the configured `results_dir`.

7. Generate the combined comparison visuals:

```bash
python src/evaluate.py --config configs/config.yaml --model all
python src/evaluate.py --config configs/config_post_dedup.yaml --model all
python src/plot_model_results.py --config configs/config.yaml
```

8. Review duplicate audit outputs and report-ready visuals:

```bash
python src/deduplicate_processed_dataset.py
```

This writes evidence files such as:

- `outputs/results/near_duplicate_examples.png`
- `outputs/results/pairwise_phash/train_val_examples.png`
- `outputs/results/pairwise_phash/train_test_examples.png`
- `outputs/results/pairwise_phash/val_test_examples.png`
- `outputs_post_dedup/results/` for the cleaned-data evaluation set

9. Run Grad-CAM on a trained checkpoint if explainability is desired:

```bash
python src/gradcam.py --config configs/config.yaml --model efficientnet_b0
```

This optional enhancement generates overlaid class-activation heatmaps saved as:

- `outputs/results/gradcam_WBC_correct.png`
- `outputs/results/gradcam_RBC_correct.png`
- `outputs/results/gradcam_Platelet_correct.png`

The same files are also copied into `outputs/report_images/` for final reporting. The overlay blending uses a visible Jet-style heatmap on the original crop so the attention regions are clearly interpretable.

## Notes

- Class imbalance handling is already built into the dataset and loss setup through weighted sampling and class-aware loss weighting.
- Checkpoint selection is driven by validation macro-F1, not raw accuracy.
- The project is implemented as a PyTorch transfer-learning pipeline rather than a Keras/TensorFlow pipeline.
- The evaluation and reporting scripts are intended to improve final reporting and comparison quality without changing the model-training behavior itself.
- The Grad-CAM script produces visible heatmap overlays for representative crop samples, and the generated images are included in `outputs/report_images/` for documentation and report assembly.

## Model comparison summary

The project contains both a baseline pre-cleanup result set and a cleaned-data retraining result set. The baseline artifacts are preserved in `outputs_pre_dedup/`, while the cleaned-data training/evaluation artifacts are stored in `outputs_post_dedup/`.

For the deduplicated dataset, the verified test distribution is:

- WBC: 127
- RBC: 1,593
- Platelet: 48
- Total: 1,768

Verified cleaned-data results are:

| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| EfficientNet-B0 | 0.997172 | 0.989587 | 0.999987 |
| MobileNetV3-Small | 0.999434 | 0.995184 | 1.000000 |
| DenseNet121 | 0.998303 | 0.992360 | 0.999996 |

The best post-dedup model is MobileNetV3-Small, with the strongest macro-F1 and near-perfect AUC on the cleaned test set. The original pre-cleanup outputs remain available for comparison and should not be confused with the cleaned-data retraining results.

The final report should use the model with the strongest macro-F1 value as the primary recommendation, while using accuracy and efficiency as secondary checks.
