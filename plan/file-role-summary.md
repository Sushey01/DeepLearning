# File Role Summary

This document describes what each main file in the project does.

## Root files

### README.md
Describes the project overview, setup instructions, environment requirements, and the workflow for running the blood-cell classification pipeline.

### requirements.txt
Lists the Python packages required to run the project, including PyTorch, torchvision, scikit-learn, matplotlib, seaborn, and related dependencies.

### configs/config.yaml
Stores all major configuration values: dataset paths, class names, image size, augmentation settings, training hyperparameters, and model choices.

## Source code files

### src/crop_dataset.py
Converts raw microscopy images and YOLO bounding-box annotations into cropped image samples organized by class and train/val/test split.

### src/dataset.py
Builds the PyTorch dataset and dataloaders. It contains image transforms, class imbalance handling, and sampler logic used during training.

### src/model.py
Defines the neural network backbones used in the project, including EfficientNet-B0, MobileNetV3-Small, and DenseNet121.

### src/losses.py
Defines the training loss functions. It handles class-weighted cross-entropy and focal loss so minority classes are not ignored because of dataset imbalance.

### src/train.py
Runs the training loop for one model at a time, tracks validation performance, saves the best checkpoint, and logs metrics to CSV.

### src/evaluate.py
Loads the saved best model and evaluates it on the held-out test set. It prints the classification report and saves the confusion matrix and JSON metrics.

### src/utils.py
Contains shared helper functions such as config loading, random seeding, and device selection for CPU/GPU.

## Output folders

### outputs/checkpoints/
Stores the best saved model weights for each trained backbone.

### outputs/logs/
Stores CSV training logs with epoch-wise loss and validation F1 values.

### outputs/results/
Stores evaluation reports and confusion matrices for each model.

## Data folders

### data/raw/
Contains the original TXL-PBC dataset files, including raw images and YOLO label files.

### data/processed/
Contains the cropped dataset after preprocessing, split into train, val, and test folders by class.

## Plan folder

### plan/project-completion-plan.md
Provides the execution plan and final project workflow, including training, evaluation, optional comparison runs, and backup strategy.

### plan/file-role-summary.md
This summary document explaining what each project file does.
