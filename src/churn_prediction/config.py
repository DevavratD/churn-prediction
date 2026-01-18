from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

MODEL_PATH = f"{PROJECT_ROOT}/models/churn_pipeline.joblib"
SCHEMA_PATH = f"{PROJECT_ROOT}/models/feature_meta.json"
BACKGROUND_SAMPLE_SIZE = 200
THRESHOLD = 0.25
RAW_DATA_PATH = f"{PROJECT_ROOT}/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
