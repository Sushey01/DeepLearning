# Dataset Reference

## Source used in the proposal

The proposal uses the TXL-PBC dataset, a curated and re-annotated peripheral blood cell dataset that integrates multiple public sources: BCCD, BCDD, PBC, and Raabin-WBC.

Official repository:
https://github.com/lugan113/TXL-PBC_Dataset

## Verification

The link was checked successfully from this environment and returned HTTP 200, confirming the repository is accessible.

## Dataset purpose

This dataset is intended for blood cell classification across:
- RBC
- WBC
- Platelet

The project pipeline crops single-cell patches from the raw microscopy images and then trains a transfer-learning classifier on the cropped data.

## Expected project usage

1. Download the TXL-PBC dataset into the project data folder as described in the repository.
2. Place images under `data/raw/images/`.
3. Place YOLO-format label files under `data/raw/labels/`.
4. Run the cropping pipeline with:
   ```bash
   python src/crop_dataset.py --config configs/config.yaml
   ```
5. Train and evaluate using the project scripts in the `src/` folder.

## Reference note

This dataset is the one cited in the proposal and matches the project objective described in the report.
