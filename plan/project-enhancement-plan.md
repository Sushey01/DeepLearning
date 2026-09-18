# Project Enhancement Plan

This file records the improvements that were either completed for the current project or remain as optional future work.

## Goal

Improve the reporting quality, validation strength, and explainability of the blood-cell classification project without changing the underlying objective of classifying WBC, RBC, and Platelet cell crops.

## Completed enhancement work

The following items have already been implemented and verified in the current repo:

### Task 1: Model comparison and metrics summary
Completed.
- The project now generates a comparison table with Macro-F1, Macro-AUC, Accuracy, parameter count, training time, and peak GPU memory.
- The data is saved under `outputs/results/comparison_table.json`.

### Task 2: Combined confusion matrix comparison figure
Completed.
- A side-by-side confusion-matrix figure is generated under `outputs/results/confusion_matrices_all.png`.

### Task 3: Loss-curve visualisation
Completed.
- Training and validation loss curves are generated for the three backbones and saved as `outputs/results/loss_curves.png`.

### Task 4: Explainability through Grad-CAM
Completed.
- The Grad-CAM script was corrected and now produces visible heatmap overlays on original cell crops.
- Outputs are saved under `outputs/results/gradcam_WBC_correct.png`, `gradcam_RBC_correct.png`, and `gradcam_Platelet_correct.png`.

### Task 5: Data quality review and deduplication audit
Completed.
- Near-duplicate image review was performed using perceptual hash checks across train/val/test pairs.
- The summary and example visuals are stored under `outputs/results/`.

### Task 6: Report-ready image package
Completed.
- The key visuals were copied into `outputs/report_images/` for easier final-report assembly.

## Remaining optional future work

These are not required to finish the current project, but they would improve the tool further for broader use.

### Task 7: Folder-based prediction script
- Add a script that takes a folder of images and saves a CSV of image name, predicted class, and confidence.

### Task 8: Confidence threshold / uncertainty flag
- Mark low-confidence predictions as uncertain instead of forcing a class label.

### Task 9: Lightweight frontend demo
- Create a simple Streamlit or Flask interface for upload and prediction.

### Task 10: External validation review
- Test on additional real-world cropped images beyond the TXL-PBC dataset to assess generalization.

### Task 11: More polished reporting UI
- Add a cleaner final webpage or markdown summary for viva/demo presentation.

## Recommended priority order for future work

1. Folder-based prediction script
2. Confidence threshold handling
3. Simple demo app
4. External validation on additional images
5. Additional clinical-style reporting polish

## Final recommendation

The project has already reached a strong completion point for the assignment. The enhancements above are now mostly optional refinements for future usability and deployment readiness, rather than required fixes for the current project.
