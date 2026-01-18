import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from churn_prediction.inference import predict_proba
from churn_prediction.decision_engine import retention_decision
from churn_prediction.shap_utils import explain
from churn_prediction.config import PROJECT_ROOT


# -------------------------------------------------
# PAGE CONFIG + GLOBAL CSS (theme-aware)
# -------------------------------------------------

st.set_page_config(page_title="Churn Decision System", layout="wide")

# CSS to fix card visibility in dark/light modes
st.markdown("""
<style>
.card {
    background-color: var(--secondary-background-color);
    padding: 12px;
    margin-bottom: 8px;
    border-radius: 8px;
    color: var(--text-color);
    font-size: 15px;
    border: 1px solid rgba(128, 128, 128, 0.2);
}
.card b {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("📉 Customer Churn Decision System")
st.caption("Predict churn risk, decide business action, and explain the prediction.")


# -------------------------------------------------
# Load sample customers
# -------------------------------------------------

SAMPLE_PATH = f"{PROJECT_ROOT}/data/sample/sample.csv"

@st.cache_data
def load_samples():
    try:
        return pd.read_csv(SAMPLE_PATH)
    except:
        return None

sample_df = load_samples()


# -------------------------------------------------
# LAYOUT
# -------------------------------------------------

left, right = st.columns([1, 1])


# =================================================
# LEFT COLUMN — INPUTS
# =================================================

with left:
    st.header("Customer Input")

    # -------------------------------
    # INPUT MODE SELECTOR
    # -------------------------------
    input_mode = st.radio(
        "Input Mode",
        ["Select Sample Customer", "Enter Manually"],
        horizontal=True
    )

    input_df = None

    # -------------------------------
    # SAMPLE MODE
    # -------------------------------
    if input_mode == "Select Sample Customer":
        if sample_df is None:
            st.warning("No sample data found.")
        else:
            df = sample_df.copy()

            # Auto-create customer_name if missing
            if "customer_name" not in df.columns:
                df["customer_name"] = df["customerID"]

            # Features passed to model
            drop_cols = ["customerID", "Churn", "TotalCharges"]
            feature_cols = [c for c in df.columns if c not in drop_cols]

            selected = st.selectbox(
                "Choose a sample customer",
                df["customer_name"].tolist()
            )

            row = df[df["customer_name"] == selected].reset_index(drop=True)

            # -------------------------------
            # SAMPLE PREVIEW — Cards (theme-aware)
            # -------------------------------
            st.subheader("Sample Customer Details")

            for col in row.columns:
                if col in ["customer_name", "Churn", "TotalCharges"]:
                    continue

                value = row[col].values[0]

                st.markdown(
                    f"""
                    <div class="card">
                        <b>{col}</b>: {value}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Only send model-required features
            input_df = row[feature_cols]

    # -------------------------------
    # MANUAL ENTRY MODE
    # -------------------------------
    else:
        st.subheader("Manual Entry")

        input_df = pd.DataFrame([{
            "gender": st.selectbox("Gender", ["Male", "Female"]),
            "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
            "Partner": st.selectbox("Partner", ["Yes", "No"]),
            "Dependents": st.selectbox("Dependents", ["Yes", "No"]),
            "tenure": st.number_input("Tenure (months)", min_value=0, value=5),
            "PhoneService": st.selectbox("Phone Service", ["Yes", "No"]),
            "MultipleLines": st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            ),
            "InternetService": st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            ),
            "OnlineSecurity": st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            ),
            "OnlineBackup": st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            ),
            "DeviceProtection": st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            ),
            "TechSupport": st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            ),
            "StreamingTV": st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            ),
            "StreamingMovies": st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            ),
            "Contract": st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            ),
            "PaperlessBilling": st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            ),
            "PaymentMethod": st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            ),
            "MonthlyCharges": st.number_input(
                "Monthly Charges", min_value=0.0, value=70.0
            )
        }])

    predict_btn = st.button("Predict", type="primary")


# =================================================
# RIGHT COLUMN — OUTPUTS
# =================================================

with right:
    st.header("Prediction Output")

    if predict_btn:
        with st.spinner("Running model..."):

            prob = predict_proba(input_df)[0]
            decision = retention_decision(prob, input_df.iloc[0])

            # -------------------------------
            # RISK METRIC
            # -------------------------------
            st.subheader("Churn Risk")
            st.metric("Probability", f"{prob:.2%}")

            # -------------------------------
            # DECISION
            # -------------------------------
            st.subheader("Recommended Action")
            st.success(decision)

            # -------------------------------
            # SHAP Explanation
            # -------------------------------
            st.subheader("Why this decision?")

            shap_values = explain(input_df)

            shap.plots.waterfall(shap_values[0], max_display=10)
            st.pyplot(plt.gcf(), bbox_inches="tight")
            plt.clf()

    else:
        st.info("Select a sample or enter customer details, then click Predict.")
