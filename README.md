# Medical Image Classification with Transfer Learning and Grad-CAM

Binary classification of chest X-rays (Normal vs. Pneumonia) using a pretrained EfficientNet-B0, fine-tuned on the target dataset, with Grad-CAM used afterward to check what the model is actually looking at when it makes a prediction.

**This is a portfolio / educational project, not a diagnostic tool. It is not intended for clinical use, and nothing here should be treated as medical advice.**

## Overview

The goal was to build a solid transfer-learning image classifier end to end — data loading, preprocessing, a two-phase training setup, evaluation, error analysis, and Grad-CAM — rather than just reporting a final accuracy number. Class imbalance, the train/val gap, and where the model actually fails are all looked at explicitly.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), originally from Kermany et al., hosted on Kaggle. Standard `train` / `val` / `test` split with two classes, `NORMAL` and `PNEUMONIA`.

The training set is noticeably imbalanced toward PNEUMONIA, which is called out explicitly rather than treated as a footnote — it's part of why recall and per-class F1 are weighted more heavily than raw accuracy when reading the results.

## Preprocessing

- Images resized and normalized using standard ImageNet mean/std, since the model starts from ImageNet-pretrained weights.
- Augmentation is kept light — a horizontal flip and a small rotation. Nothing aggressive like color jitter, since these are grayscale medical images and heavier augmentation risks introducing artifacts that don't correspond to anything real in an X-ray.

## Model: EfficientNet-B0 (Transfer Learning)

Starts from ImageNet-pretrained weights, with the classifier head replaced for the 2-class problem.

**Phase 1 — head only.** The feature extractor is frozen so the pretrained weights aren't disturbed early on; only the new classifier head trains.

**Phase 2 — fine-tuning.** Only the last block of EfficientNet-B0 is unfrozen (not the whole network), with a lower learning rate. Earlier layers already encode generic, transferable features (edges, textures) from ImageNet, so those are left alone — unfreezing everything with a normal learning rate risks the first few noisy batches damaging those weights before the model settles.

## Evaluation Metrics

- Accuracy, Precision, Recall, F1 (on the PNEUMONIA class)
- ROC-AUC, using the predicted probability of PNEUMONIA
- Confusion matrix on the held-out test set

The test set is only touched once, at the end, using the checkpoint with the best validation accuracy.

## Final Results

| Metric | Value |
|---|---|
| Accuracy | 87.0% |
| Precision | 84.9% |
| Recall | 96.4% |
| F1 | 90.3% |
| ROC-AUC | 94.6% |

Test set: 624 images — 81 total misclassifications (67 false positives, 14 false negatives).

Recall on PNEUMONIA is prioritized over raw accuracy in how these results are read, since missing an actual pneumonia case (a false negative) is generally a worse mistake than a false alarm. That framing isn't a clinical claim — it's just the lens used to decide which errors matter more when interpreting the numbers.

## Grad-CAM

Grad-CAM (via the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library) highlights the pixels that most influenced the model's prediction for a given image. It's used here purely as a sanity check — the overlays show the model attending to areas inside the lung region rather than borders, text markers, or other irrelevant parts of the scan.

Grad-CAM does **not** prove the model is using medically meaningful features. It only shows where the network was looking, not why, and it isn't a substitute for clinical validation.

## Error Analysis

False positives and false negatives from the test set were pulled out and inspected directly, rather than only looking at the confusion matrix totals. Some of the false negatives visually resemble the correctly classified NORMAL images, which raises the possibility that the model is partly picking up on overall image brightness/contrast rather than a more localized, specific signal — Grad-CAM was used as a rough check on that concern, though it can't fully confirm or rule it out.

## Limitations

- Trained and evaluated on a single dataset/source — results may not generalize to X-rays from a different hospital or scanner.
- Binary classification only (Normal vs. Pneumonia); doesn't distinguish bacterial vs. viral pneumonia or other conditions.
- The backbone is ImageNet-pretrained, which is a substantial domain gap from chest X-rays.
- No external, held-out dataset from a different source was used to check generalization.
- Grad-CAM is an interpretability aid, not proof the model is using clinically valid features.
- This is a portfolio/learning project, not a diagnostic tool, and isn't intended for clinical use.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `notebooks/medical_image_classification.ipynb` in Jupyter or Colab. GPU is strongly recommended for training.
3. The dataset is pulled from Kaggle via `kagglehub`, which requires a Kaggle account and API token:
   - Go to kaggle.com → Account settings → "Create New API Token" to download `kaggle.json`.
   - Follow the upload prompt in the notebook's dataset-loading cell.
   - **Do not commit `kaggle.json` to this or any repository** — it's already excluded via `.gitignore`.
4. Run all cells top to bottom.

## Project Structure

```
medical-image-classification/
├── README.md
├── notebooks/
│   └── medical_image_classification.ipynb
├── figures/
│   ├── training_loss.png
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   └── gradcam_example.png
├── results/
│   └── medical_image_results.csv
├── requirements.txt
└── .gitignore
```

Note: the notebook also generates a class-distribution chart and an ROC curve chart, but those two files weren't available locally when this repo was packaged up, so they're left out rather than faked. Re-running the notebook will regenerate them into `figures/`. Trained model weights (`best_efficientnet_b0.pth`) are intentionally not included in this repo — re-run the notebook to reproduce them.
