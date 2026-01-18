import joblib
import json

from churn_prediction.config import MODEL_PATH,SCHEMA_PATH


def load_artifacts():
    clf = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    return clf, schema

def predict_proba(raw_df):
    clf, _ = load_artifacts()
    return clf.predict_proba(raw_df)[:, 1]