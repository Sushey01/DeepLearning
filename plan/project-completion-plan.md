# Project Completion Plan

## Current status

The pipeline is already functional and the repo is in a strong state for finishing the project:

- The workspace has no reported Python errors.
- The dataset has already been processed and class folders exist under `data/processed/train`, `data/processed/val`, and `data/processed/test`.
- A batch-level smoke test confirmed that the EfficientNet-B0 model can load and process data successfully.
- The project is now being narrowed to a practical final delivery plan, rather than broad benchmarking for every model.

## Key decision

The main model for this project will be EfficientNet-B0. The other backbones are optional comparison experiments, not mandatory deliverables.

This is the right choice because:
- it is already the preferred backbone in the repository
- it is a strong transfer-learning model for this task
- it keeps the project realistic and finishable within time

## Phase 1: Final data verification

- Confirm that the required TXL-PBC files are present under `data/raw/images` and `data/raw/labels`.
- Re-check class counts after cropping and verify the distribution is reasonable.
- Ensure the split files are aligned and there is no data leakage between train/val/test.
- Verify that the class ordering in `configs/config.yaml` matches the YOLO labels.

## Phase 2: Main training run (priority)

Run the primary training job in a long-running terminal so it is not interrupted:

```bash
source venv/bin/activate
python src/train.py --config configs/config.yaml --model efficientnet_b0
```

Checklist:
- checkpoint saved in `outputs/checkpoints/`
- logs written to `outputs/logs/`
- best model chosen on validation macro-F1
- training time and memory recorded

## Phase 3: Primary evaluation

Run the evaluation for the chosen main model:

```bash
python src/evaluate.py --config configs/config.yaml --model efficientnet_b0
```

Collect:
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- per-class recall for WBC and Platelet

## Phase 4: Optional comparison run

Only do this if the assignment or supervisor expects a model comparison.

Run MobileNetV3-Small as a lightweight comparison:

```bash
python src/train.py --config configs/config.yaml --model mobilenet_v3_small
python src/evaluate.py --config configs/config.yaml --model mobilenet_v3_small
```

DenseNet121 is optional and should only be trained if time allows.

## Phase 5: Backup and transfer workflow

Because checkpoints and logs are not tracked in GitHub, store them outside the repo when training is complete.

Recommended workflow:
- keep `outputs/` in the project for active use
- zip or copy `outputs/checkpoints`, `outputs/logs`, and `outputs/results` to Google Drive later
- do not commit these files to GitHub

Example:

```powershell
Compress-Archive -Path .\outputs\* -DestinationPath .\project_outputs_backup.zip -Force
```

Then later move the zip to Google Drive.

## Phase 6: Final report

Prepare the final write-up with:
- dataset overview and preprocessing
- class imbalance handling strategy
- EfficientNet-B0 model design and training process
- evaluation metrics and confusion matrix interpretation
- final recommendation and limitations

## Final recommendation

The project should be completed around the EfficientNet-B0 pipeline first. The extended comparison across all three models is optional and should only be done if the task specifically requires it.

This keeps the work realistic, reduces wasted training time, and still gives a strong final result.

## Immediate next step

The next action is to train the EfficientNet-B0 model, evaluate it on the held-out test set, and only then decide whether any comparison runs are necessary.
