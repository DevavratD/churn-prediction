import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import json


from churn_prediction.pipeline import build_pipeline, NUM_FEATURES, CAT_FEATURES
from churn_prediction.config import MODEL_PATH, SCHEMA_PATH,RAW_DATA_PATH


def train():
    df = pd.read_csv(RAW_DATA_PATH)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    joblib.dump(clf, MODEL_PATH)

    preprocessor = clf.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CAT_FEATURES)
    feature_schema = {
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "transformed_features": NUM_FEATURES + list(cat_names),
    }

    with open(SCHEMA_PATH, "w") as f:
        json.dump(feature_schema, f, indent=2)

    print("Model and feature meta saved")


if __name__ == "__main__":
    train()
