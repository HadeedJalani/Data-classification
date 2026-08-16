# Project 2 — Data Classification Using AI

Iris classification internship project implementing the supplied assignment.

## Pipeline

Iris CSV → validation → stratified 80/20 split → StandardScaler → KNN → prediction → confusion matrix + macro F1

- Dataset: 150 samples, 3 classes, 4 numerical features
- Split: 80/20 stratified, `random_state=42`
- Algorithm: K-Nearest Neighbors
- K selection: 5-fold cross-validation over K=1..15 using macro F1
- Demonstration K: 5
- Evaluation: accuracy, macro precision, macro recall, macro F1, confusion matrix

## Verified result

```text
Accuracy: 0.9333
Macro Precision: 0.9444
Macro Recall: 0.9333
Macro F1: 0.9327
Confusion Matrix:
[10, 0, 0]
[0, 10, 0]
[0, 2, 8]
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python -m src.main
python -m src.main --no-tune-k --k 5
pytest -q
```

## Project structure

```text
Data-classification/
├── data/iris.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   └── visualize.py
├── tests/test_pipeline.py
├── artifacts/        # generated locally; not required in Git
├── reports/          # generated locally; not required in Git
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Architecture

1. Load and validate `data/iris.csv`.
2. Shuffle and create a stratified 80/20 train/test split.
3. Fit `StandardScaler` only on training data through a scikit-learn `Pipeline`.
4. Select K using 5-fold cross-validation and macro F1 unless `--no-tune-k` is supplied.
5. Fit KNN and predict the held-out test set.
6. Save metrics, classification report, confusion matrix, and K-selection plot.

## Demo

For the exact assignment demonstration:

```powershell
python -m src.main --no-tune-k --k 5
```

Expected headline result: 93.33% accuracy and 0.9327 macro F1 on the 30-sample test set.

## Scope

The supplied slides mention computer vision/CNNs as future directions. They are not part of this Iris classification implementation, so no unnecessary deep-learning stack was added.
