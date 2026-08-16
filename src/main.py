from __future__ import annotations

import argparse
import json

import joblib

from .config import ARTIFACTS_DIR, DATA_PATH, DEFAULT_K, REPORTS_DIR
from .data import load_dataset, split_dataset
from .evaluate import evaluate_model
from .model import build_model, select_k
from .visualize import save_confusion_matrix, save_k_selection


def parse_args():
    parser = argparse.ArgumentParser(description="Project 2 - Data Classification Using AI")
    parser.add_argument("--no-tune-k", action="store_true", help="Skip CV tuning and use --k")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="K for KNN when tuning is disabled")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)

    if args.no_tune_k:
        selected_k = args.k
        scores = {}
    else:
        selected_k, scores = select_k(X_train, y_train)
        save_k_selection(scores, ARTIFACTS_DIR / "k_selection.png", selected_k)

    model = build_model(selected_k)
    model.fit(X_train, y_train)
    metrics, _ = evaluate_model(model, X_test, y_test, ARTIFACTS_DIR, REPORTS_DIR)
    metrics["selected_k"] = selected_k
    metrics["dataset"] = str(DATA_PATH)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, ARTIFACTS_DIR / "iris_knn_model.joblib")
    save_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], ARTIFACTS_DIR / "confusion_matrix.png")

    print("Project 2 - Data Classification Using AI")
    print(f"Dataset: {DATA_PATH}")
    print(f"Samples: {len(df)} | Classes: {df['species'].nunique()} | Features: 4")
    print(f"Class distribution: {df['species'].value_counts().sort_index().to_dict()}")
    print(f"Train/Test: {len(X_train)}/{len(X_test)} (80/20)")
    print(f"Selected K: {selected_k}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print("Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(row)


if __name__ == "__main__":
    main()
