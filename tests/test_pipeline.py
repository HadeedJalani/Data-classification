from pathlib import Path

from sklearn.metrics import f1_score

from src.config import DATA_PATH
from src.data import load_dataset, split_dataset
from src.model import build_model, select_k


def test_dataset_schema_and_shape():
    df = load_dataset(DATA_PATH)
    assert df.shape == (150, 5)
    assert df["species"].nunique() == 3
    assert df["species"].value_counts().to_dict() == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_stratified_split():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert y_train.value_counts().to_dict() == {"setosa": 40, "versicolor": 40, "virginica": 40}
    assert y_test.value_counts().to_dict() == {"setosa": 10, "versicolor": 10, "virginica": 10}


def test_model_pipeline():
    model = build_model(5)
    assert "scaler" in model.named_steps
    assert "knn" in model.named_steps
    assert model.named_steps["knn"].n_neighbors == 5


def test_k_selection_is_valid():
    df = load_dataset(DATA_PATH)
    X_train, _, y_train, _ = split_dataset(df)
    k, scores = select_k(X_train, y_train)
    assert 1 <= k <= 15
    assert len(scores) == 15


def test_end_to_end_quality():
    df = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(df)
    model = build_model(5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert f1_score(y_test, predictions, average="macro") >= 0.90
