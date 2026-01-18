import json
import joblib
import shap
import numpy as np
import pandas as pd

from churn_prediction.config import MODEL_PATH, SCHEMA_PATH


# ------------------------------------------------------
# Load artifacts (pipeline + schema)
# ------------------------------------------------------
def _load_artifacts():
    clf = joblib.load(MODEL_PATH)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    # Flexible key detection (schema evolution)
    if "transformed_features" in schema:
        feature_names = schema["transformed_features"]
    elif "feature_names" in schema:
        feature_names = schema["feature_names"]
    else:
        raise KeyError(f"Schema missing feature names. Found keys: {list(schema.keys())}")

    return clf, feature_names


# ------------------------------------------------------
# Build SHAP explainer (NO RAW DATA)
# ------------------------------------------------------
_explainer_cache = None

def _build_explainer():
    global _explainer_cache
    if _explainer_cache is not None:
        return _explainer_cache

    clf, feature_names = _load_artifacts()

    preprocessor = clf.named_steps["preprocessor"]
    linear_model = clf.named_steps["model"]

    # ----------------------------------------------
    # GENERATE SYNTHETIC BACKGROUND
    # ----------------------------------------------
    # This is correct for LogisticRegression + StandardScaler:
    # Scaled features → centered around 0
    background = np.zeros((50, len(feature_names)))

    explainer = shap.LinearExplainer(
        linear_model,
        background,      # cloud-safe baseline
        feature_names=feature_names
    )

    _explainer_cache = (explainer, feature_names, preprocessor)
    return _explainer_cache


# ------------------------------------------------------
# Explain a single sample
# ------------------------------------------------------
def explain(raw_df: pd.DataFrame):
    explainer, feature_names, preprocessor = _build_explainer()

    # Transform with pipeline
    X_trans = preprocessor.transform(raw_df)

    # Wrap into DF for feature names
    X_trans = pd.DataFrame(X_trans, columns=feature_names)

    shap_values = explainer(X_trans)

    # SAFETY — schema alignment
    assert shap_values.values.shape[1] == len(feature_names), \
        "Feature schema mismatch — SHAP aborted"

    return shap_values
