# Project Enhancement Plan

This file outlines the practical next improvements for the current blood-cell classification project.

## Goal

Improve the usability, reporting quality, and validation strength of the project without changing the overall objective of classifying WBC, RBC, and Platelet cell crops.

## Phase 1: Improve inference usability

### Task 1: Add folder-based prediction
- Create a script that takes a folder of images and predicts each one.
- Save results to a CSV file containing:
  - image name
  - predicted class
  - confidence
- This is useful for quick model testing on multiple images.

### Task 2: Add a confidence threshold
- If prediction confidence is below a chosen threshold, mark it as uncertain.
- This helps avoid forcing a label when the model is unsure.
- Useful for clinical-style interpretation and safer decision making.

### Task 3: Improve the terminal output
- Print the prediction in a cleaner format.
- Show the top 3 class probabilities.
- Add optional verbose mode for debugging.

## Phase 2: Add a lightweight frontend

### Task 4: Create a simple Streamlit app
- Upload one image or a folder of images.
- Display the prediction in the browser.
- Show confidence and class probabilities.
- This turns the project into a demo-ready tool.

### Task 5: Optional Flask app
- If needed, create a lightweight web interface for a more traditional web app setup.
- Good for demonstrating the model to a supervisor or project audience.

## Phase 3: Strengthen validation

### Task 6: Test on more real cropped samples
- Use additional cropped images from the same dataset or external validation set.
- Check if predictions remain stable across different image sources.

### Task 7: Add external validation review
- Compare model performance on known external cell images.
- Identify if the model is domain-specific to the TXL-PBC style.

### Task 8: Review class imbalance handling
- Reassess whether the current augmentation and weighting are enough.
- Check whether Platelet and WBC recall can be improved further.

## Phase 4: Improve reporting quality

### Task 9: Add a model comparison table
- Include accuracy, macro F1, parameter count, training time, and memory use.
- Use this as the final report table.

### Task 10: Add richer result plots
- Create bar charts for per-class recall and F1.
- Save confusion matrices for each model in a cleaner final-output format.

### Task 11: Add a final project summary page
- Prepare a short markdown or report section summarizing:
  - dataset
  - preprocessing
  - models compared
  - best model
  - final recommendation

## Phase 5: Optional advanced improvements

### Task 12: Add explainability
- Use Grad-CAM or similar tools to highlight which image region influenced the decision.
- Helpful for understanding whether the model is focusing on the actual cell rather than background noise.

### Task 13: Add a live demo workflow
- Prepare a presentation-ready demo where a user uploads a cell image and sees the prediction instantly.
- This is useful for viva or project demonstration.

## Recommended priority order

1. Folder-based prediction script
2. Confidence threshold and uncertainty handling
3. Simple Streamlit demo
4. Better validation on more sample images
5. Final report and comparison table improvements

## Final recommendation

The project is already strong enough to finish as a working blood-cell classification system. The enhancements above are best used to make it more usable, more polished, and more convincing in a report or demo, rather than as required fixes.
