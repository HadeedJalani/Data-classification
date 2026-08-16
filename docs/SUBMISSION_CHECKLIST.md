# Project 2 Submission Checklist

## Required / assignment-aligned

- Source code under `src/`
- Iris dataset under `data/iris.csv`
- `requirements.txt`
- README and setup instructions
- StandardScaler preprocessing
- 80/20 stratified train/test split
- KNN classifier
- K selection / fixed K=5 demonstration
- Confusion matrix
- Macro precision, recall and F1
- Automated tests

## Recommended for supervisor

- Run `pytest -q`
- Run `python -m src.main`
- Run `python -m src.main --no-tune-k --k 5`
- Show the generated confusion matrix and K-selection plot
- Explain the two Virginica → Versicolor errors
- Use the supplied internship presentation/report as presentation material
