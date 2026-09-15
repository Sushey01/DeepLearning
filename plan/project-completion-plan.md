# Project Completion Plan

## Current status

The project pipeline is already in a workable state:

- The editor reports no Python errors for the workspace.
- The training data generation step has produced class folders under `data/processed/train`, `data/processed/val`, and `data/processed/test`.
- A smoke test confirmed that a single training batch can be loaded and passed through the EfficientNet-B0 model successfully.

The only issue observed so far is operational: the full training run is long-running and appears to time out under a short terminal limit, but it is not failing with a code-level exception during the initial forward/backward pass.

## Phase 1: Dataset verification

- Confirm the TXL-PBC data is present under `data/raw/images` and `data/raw/labels`.
- Re-check the class counts after crop generation and keep them aligned with the expected RBC/WBC/Platelet distribution.
- Ensure the split files match the image-level assignments and do not leak the same original image across train/val/test.

## Phase 2: Training validation

Run the model pipeline in a longer-lived terminal session so the job is not interrupted by timeout limits:

```bash
source venv/bin/activate
python src/train.py --config configs/config.yaml --model efficientnet_b0
```

Then repeat for the other candidate backbones:

```bash
python src/train.py --config configs/config.yaml --model mobilenet_v3_small
python src/train.py --config configs/config.yaml --model densenet121
```

Checklist:
- model checkpoint saved in `outputs/checkpoints/`
- CSV metrics logged in `outputs/logs/`
- early stopping triggered only when validation macro-F1 stops improving
- best checkpoint chosen on validation macro-F1, not raw accuracy

## Phase 3: Evaluation and comparison

Once each model finishes training, run the evaluator:

```bash
python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
python src/evaluate.py --config configs/config.yaml --model mobilenet_v3_small
python src/evaluate.py --config configs/config.yaml --model densenet121
```

Collect:
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrices
- per-class recall for WBC and Platelet

## Phase 4: Final model selection

- Compare the three models using the saved evaluation JSON outputs.
- Select the strongest model based on validation macro-F1 and the final test-set macro F1.
- Use the WBC/Platelet recall values as the clinically important decision metric for the final recommendation.

## Phase 5: Report completion

Prepare the final project write-up with:
- dataset description
- preprocessing and split strategy
- model setup and training choices
- evaluation results and confusion matrix interpretation
- final recommendation and limitations

## Immediate next step

The next action is to let the full training run finish in the background for the selected backbone, then save the evaluation artifacts and compare them before writing the final report.
