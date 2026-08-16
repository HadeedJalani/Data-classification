# Project 2 — Technical Report

## Objective

Build a basic supervised classification model on the Iris dataset using the assignment workflow: load and understand data, split into training/testing sets, scale features, apply KNN, and evaluate with a confusion matrix and F1 score.

## Dataset

- Samples: 150
- Classes: 3
- Features: 4
- Distribution: 50 setosa, 50 versicolor, 50 virginica

## Architecture

```text
CSV
 ↓
Validation
 ↓
Shuffle + Stratified 80/20 Split
 ↓
StandardScaler
 ↓
K-Nearest Neighbors
 ↓
Prediction
 ↓
Accuracy / Precision / Recall / Macro F1 / Confusion Matrix
```

## Model

The classifier is a scikit-learn `Pipeline` containing `StandardScaler` followed by `KNeighborsClassifier`. The scaler is fitted only from training data, preventing test-set leakage.

K tuning evaluates K=1..15 with 5-fold stratified cross-validation and macro F1. The assignment's reproducible demonstration explicitly uses K=5.

## Verified K=5 result

- Accuracy: 0.9333
- Macro Precision: 0.9444
- Macro Recall: 0.9333
- Macro F1: 0.9327
- Confusion Matrix: `[[10, 0, 0], [0, 10, 0], [0, 2, 8]]`

Interpretation: setosa and versicolor were classified perfectly in the held-out set; two virginica samples were predicted as versicolor.

## Testing

The repository includes five automated tests covering dataset shape/distribution, stratified splitting, model construction, K selection, and end-to-end quality.

Run:

```powershell
pytest -q
```

## Demo

```powershell
python -m src.main --no-tune-k --k 5
```

## Scope

Computer vision and CNNs shown as future directions in the assignment slides are intentionally outside this Project 2 implementation.
