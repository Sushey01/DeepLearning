# Post-Deduplication Model Results

## Dataset

The models were retrained and evaluated on the cleaned processed dataset.

| Split | WBC | RBC | Platelet | Total |
|---|---:|---:|---:|---:|
| Train | 901 | 10,854 | 382 | 12,137 |
| Validation | 252 | 3,180 | 112 | 3,544 |
| Test | 127 | 1,593 | 48 | 1,768 |

The deduplication cleanup removed 694 duplicate images from the original processed dataset.

## New Model Results

| Model | Training time (sec) | Evaluation time (sec) | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 1,396.73 | 10.58 | 0.997172 | 0.989587 | 0.999987 |
| MobileNetV3-Small | 895.60 | 8.70 | 0.999434 | 0.995184 | 1.000000 |
| DenseNet121 | 1,448.15 | 12.14 | 0.998303 | 0.992360 | 0.999996 |

## Post-Dedup Training and Evaluation

| Model | Training time | Evaluation time | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 1,396.73 sec | 10.58 sec | 0.997172 | 0.989587 |
| MobileNetV3-Small | 895.60 sec | 8.70 sec | 0.999434 | 0.995184 |
| DenseNet121 | 1,448.15 sec | 12.14 sec | 0.998303 | 0.992360 |

## Before vs After

| Model | Old accuracy | New accuracy | Old Macro-F1 | New Macro-F1 |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 1.000000 | 0.997172 | 1.000000 | 0.989587 |
| MobileNetV3-Small | 0.998937 | 0.999434 | 0.993959 | 0.995184 |
| DenseNet121 | 0.998937 | 0.998303 | 0.993978 | 0.992360 |

## Best Model

MobileNetV3-Small achieved the best post-deduplication performance:

- Accuracy: 0.999434
- Macro-F1: 0.995184
- Macro-AUC: 1.000000
- Training time: 895.60 seconds

## Generated Artifacts

- Checkpoints: `outputs_post_dedup/checkpoints/`
- Training logs: `outputs_post_dedup/logs/`
- Individual evaluation reports: `outputs_post_dedup/results/*_report.json`
- Comparison table: `outputs_post_dedup/results/comparison_table.json`
- Individual confusion matrices: `outputs_post_dedup/results/*_confusion_matrix.png`
- Combined confusion matrices: `outputs_post_dedup/results/confusion_matrices_all.png`
- Loss curves: `outputs_post_dedup/results/loss_curves.png`

The original pre-cleanup artifacts in `outputs_pre_dedup/` were not modified.
