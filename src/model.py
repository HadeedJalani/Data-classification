from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import CV_FOLDS, K_RANGE, RANDOM_STATE


def build_model(k: int) -> Pipeline:
    if k < 1:
        raise ValueError("k must be >= 1")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def select_k(X_train, y_train) -> tuple[int, dict[int, float]]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores: dict[int, float] = {}
    for k in K_RANGE:
        model = build_model(k)
        scores[k] = float(cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro").mean())
    best_score = max(scores.values())
    best_k = min(k for k, score in scores.items() if score == best_score)
    return best_k, scores
