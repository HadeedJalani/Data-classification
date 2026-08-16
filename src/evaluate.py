from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test, output_dir: Path, report_dir: Path):
    predictions = model.predict(X_test)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    (report_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    return metrics, predictions
