# src/churn_prediction/shap_utils.py

import json
import joblib
import shap
import pandas as pd
from functools import lru_cache

from churn_prediction.config import MODEL_PATH, SCHEMA_PATH, RAW_DATA_PATH


# -------------------------------
# Internal helpers
# -------------------------------

@lru_cache(maxsize=1)
def _load_background():
    """
    Load and cache background data for SHAP.
    This runs ONCE per process.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    # sample to keep SHAP fast
    return df.sample(200, random_state=42)


@lru_cache(maxsize=1)
def _build_explainer():
    """
    Build and cache the SHAP explainer.
    """
    clf = joblib.load(MODEL_PATH)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    preprocessor = clf.named_steps["preprocessor"]
    model = clf.named_steps["model"]

    background_df = _load_background()
    X_bg_trans = preprocessor.transform(background_df)

    explainer = shap.LinearExplainer(model, X_bg_trans)

    return explainer, schema["transformed_features"], preprocessor


# -------------------------------
# Public API
# -------------------------------

def explain(raw_df: pd.DataFrame):
    """
    Generate SHAP values for a single inference row.
    """
    explainer, feature_names, preprocessor = _build_explainer()

    X_trans = preprocessor.transform(raw_df)
    shap_values = explainer(X_trans)

    # hard safety check
    if shap_values.values.shape[1] != len(feature_names):
        raise RuntimeError("SHAP feature mismatch with schema")

    shap_values.feature_names = feature_names
    return shap_values
