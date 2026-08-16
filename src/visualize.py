from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_k_selection(scores: dict[int, float], path: Path, selected_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(list(scores), list(scores.values()), marker="o")
    plt.axvline(selected_k, linestyle="--", label=f"Selected K={selected_k}")
    plt.xlabel("K")
    plt.ylabel("Mean CV Macro F1")
    plt.title("K Selection by 5-Fold Cross-Validation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix(cm, labels, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Iris KNN Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
