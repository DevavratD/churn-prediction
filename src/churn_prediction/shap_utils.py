import joblib
import shap
import pandas as pd
import numpy as np

from churn_prediction.config import MODEL_PATH, SCHEMA_PATH


# ----------------------------------------------------------
# Load artifacts only once
# ----------------------------------------------------------

def _load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    feature_names = schema["feature_names"]
    return model, feature_names


# ----------------------------------------------------------
# Build SHAP explainer WITHOUT RAW DATA (cloud-safe)
# ----------------------------------------------------------

_explainer_cache = None

def _build_explainer():
    global _explainer_cache

    if _explainer_cache is not None:
        return _explainer_cache

    model, feature_names = _load_artifacts()
    preprocessor = model.named_steps["preprocessor"]
    linear = model.named_steps["model"]

    # Create a synthetic background instead of raw data
    # This is the correct way for deployment
    background = np.zeros((50, len(feature_names)))

    explainer = shap.LinearExplainer(
        linear,
        background,
        feature_names=feature_names
    )

    _explainer_cache = (explainer, feature_names, preprocessor)
    return _explainer_cache


# ----------------------------------------------------------
# Explain a single input row
# ----------------------------------------------------------

def explain(input_df: pd.DataFrame):
    explainer, feature_names, preprocessor = _build_explainer()

    # transform input
    X_trans = preprocessor.transform(input_df)

    # Wrap into DataFrame for SHAP naming
    X_trans = pd.DataFrame(X_trans, columns=feature_names)

    shap_values = explainer(X_trans)

    return shap_values
