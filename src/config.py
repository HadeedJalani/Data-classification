from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "iris.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.20
DEFAULT_K = 5
K_RANGE = range(1, 16)
CV_FOLDS = 5
TARGET_COLUMN = "species"
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for directory in (ARTIFACTS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
