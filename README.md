# Customer Churn Prediction System

An end-to-end **machine learning system** for predicting customer churn, converting predictions into **business actions**, and explaining decisions using **SHAP**.

**Live App:**  
https://churn-prediction-g8crja4f54hwpv4lxwhcnf.streamlit.app/

---

## What this project demonstrates

This project focuses on **how ML is actually used in practice**, not just model training.

- Modular preprocessing + modeling pipeline (`ColumnTransformer`, `Pipeline`)
- Class-imbalance aware churn prediction
- Probability-based decision rules (not hard labels)
- Model explainability with SHAP
- Interactive Streamlit interface
- Cloud deployment (Streamlit Cloud)

---

## System overview

**Flow:**

Customer data → Preprocessing → Churn probability → Business decision → SHAP explanation

The model predicts *risk*, a decision layer converts risk into *action*, and SHAP explains *why*.

---

## Model

- Algorithm: Logistic Regression  
- Class imbalance handling: `class_weight="balanced"`  
- Preprocessing:
  - Numerical → imputation + scaling
  - Categorical → one-hot encoding  
- Entire pipeline saved as a single artifact

Logistic Regression was chosen for stability, interpretability, and explainability.

---

## Decision logic

Predictions are used to drive actions, not just classification.

Examples:
- High risk + month-to-month → retention discount
- Low tenure → priority retention call
- Medium risk → retention email
- Low risk → no action

This mirrors real churn-prevention systems.

---

## Explainability

SHAP is used to explain individual predictions.

- Feature-level contribution for each customer
- Waterfall plots for transparency
- Cloud-safe implementation (no raw data dependency)
- Feature alignment guaranteed via saved schema

---

## Interface

The Streamlit app allows:
- Manual customer input
- Sample customer selection
- Full customer profile preview
- Churn probability display
- Recommended action
- SHAP explanation of the prediction

---

## Project structure

churn-prediction/
├── app/ # Streamlit application
├── src/churn_prediction # ML pipeline, inference, SHAP, decision logic
├── models/ # Trained pipeline + feature schema
├── data/sample/ # Sample customers for demo
├── pyproject.toml
└── README.md


## Run locally

```bash
git clone https://github.com/DevavratD/churn-prediction.git
cd churn-prediction
uv sync
streamlit run app/streamlit_app.py
