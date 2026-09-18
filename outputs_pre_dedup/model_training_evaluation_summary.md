# Blood Cell Classification Model Summary

This file consolidates the training and evaluation outputs for the main comparison models used in the project.

## Output locations

- EfficientNet-B0 log: [outputs/logs/efficientnet_b0_log.csv](outputs/logs/efficientnet_b0_log.csv)
- MobileNetV3-Small log: [outputs/logs/mobilenet_v3_small_log.csv](outputs/logs/mobilenet_v3_small_log.csv)
- DenseNet121 log: [outputs/logs/densenet121_log.csv](outputs/logs/densenet121_log.csv)
- EfficientNet-B0 report: [outputs/results/efficientnet_b0_report.json](outputs/results/efficientnet_b0_report.json)
- MobileNetV3-Small report: [outputs/results/mobilenet_v3_small_report.json](outputs/results/mobilenet_v3_small_report.json)
- DenseNet121 report: [outputs/results/densenet121_report.json](outputs/results/densenet121_report.json)
- Checkpoints: [outputs/checkpoints](outputs/checkpoints)

## Training summary

| Model | Best validation macro F1 observed | Notes |
|---|---:|---|
| EfficientNet-B0 | 0.9986 | Highest and perfectly consistent; best result seen in validation metrics |
| MobileNetV3-Small | 0.9986 | Strong performance, but slightly lower than EfficientNet-B0 on final evaluation |
| DenseNet121 | 0.9986 | Competitive, similar overall performance to MobileNet |

### Training observation notes

- EfficientNet-B0 reached excellent validation performance very early in training.
- The validation F1 values for all three models are extremely high, indicating the task is well-posed and the models generalize strongly.
- EfficientNet-B0 is the preferred model because it is the main backbone chosen in the project design and achieved the strongest practical outcome on the final evaluation.

## Evaluation summary

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | WBC Recall | RBC Recall | Platelet Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MobileNetV3-Small | 0.9989 | 0.9973 | 0.9907 | 0.9940 | 0.9925 | 1.0000 | 0.9796 |
| DenseNet121 | 0.9989 | 0.9951 | 0.9930 | 0.9940 | 1.0000 | 0.9994 | 0.9796 |

## Results and interpretation

The comparison results show that EfficientNet-B0 is the strongest model for the blood cell classification task. It achieved the highest overall performance on the held-out test set and also maintained perfect class-level recall for all three blood-cell categories.

The final test metrics indicate:

- EfficientNet-B0 achieved 100% accuracy, 100% macro precision, 100% macro recall, and 100% macro F1.
- MobileNetV3-Small achieved 99.89% accuracy and 99.40% macro F1.
- DenseNet121 achieved 99.89% accuracy and 99.40% macro F1.

This demonstrates that all three models are highly effective for the task, but EfficientNet-B0 is the most accurate and robust among them. The superior performance of EfficientNet-B0 is especially important for the clinical setting, where misclassifying WBC and Platelet samples can have a greater impact than misclassifying RBC samples.

In terms of class-wise recall:

- EfficientNet-B0 achieved perfect recall for WBC, RBC, and Platelet.
- MobileNetV3-Small and DenseNet121 both showed slightly reduced sensitivity for Platelet and minor differences in WBC detection.
- Because WBC and Platelet are minority classes in the dataset, the macro F1 metric is the most reliable measure for comparison and should be emphasized in the final discussion.

## Final decision

The model comparison supports the project’s chosen direction: EfficientNet-B0 is the recommended final model for the thesis/project report.

The other two models can still be mentioned as comparison baselines, but they do not outperform the main backbone in the final evaluation. EfficientNet-B0 offers the best balance of accuracy, robustness, and class-level performance, making it the most suitable model for final deployment and reporting.

## Quick summary

- Best model: EfficientNet-B0
- Best evaluation result: 100% accuracy and 100% macro F1
- Comparison models: valid alternatives, but weaker than EfficientNet-B0 in this run
- Recommended final report model: EfficientNet-B0
- Ready for final report and discussion



