# Customer Churn — Exploratory Data Analysis (EDA)

## Problem Understanding

### What is churn?
**Churn refers to customers who discontinued their service during the last billing cycle.**  
Churn is observed **after some time has passed**, not at the moment of signup.

---

### Why churn prediction matters
Customer retention is significantly cheaper than acquiring new customers.  
Accurately identifying customers at high risk of churn enables proactive retention strategies.

---

### What decision does this model support?
The model supports decisions around:
- **Which customers to target for retention**
- **Where to allocate retention resources efficiently**

---

## Target Distribution

### Class imbalance
- Overall churn rate is approximately **26%**
- Non-churned customers dominate the dataset

**Implication:**  
Accuracy alone is misleading; evaluation should emphasize **ROC-AUC, Recall, Precision, and PR-AUC**.

---

## Key EDA Insights

### 1. Tenure vs Churn
- Customers who churn have **significantly lower tenure**
- Loyal customers remain active for longer durations
- Some overlap exists (≈ 15–29 months), but separation is clear overall

**Implication:**  
Tenure is a strong churn indicator but encodes **survival time**, introducing potential **leakage risk**.

---

### 2. Tenure = 0 Edge Case
- Customers with tenure = 0 have **0% churn**
- This is a **structural constraint**, not loyalty
- Churn cannot occur before a customer has existed long enough

**Implication:**  
Tenure = 0 represents **censored data**, not meaningful churn behavior.

---

### 3. Monthly Charges vs Churn
- Churned customers have a **higher median MonthlyCharges**
- Distributions **overlap heavily**
- MonthlyCharges alone cannot reliably distinguish churners

**Implication:**  
Pricing is associated with churn but must be **combined with other features**.

---

### 4. MonthlyCharges vs TotalCharges (Joint Analysis)
- Strong diagonal relationship for **non-churned customers**
- Weaker diagonal for **churned customers**
- Churned customers cluster at **low TotalCharges**, even with high MonthlyCharges

**Interpretation:**  
Churned customers exit early and do not accumulate TotalCharges.

**Implication:**  
TotalCharges largely reflects **tenure / survival time**, introducing redundancy and leakage risk.

---

### 5. Contract Type vs Churn
- Churn rate decreases sharply as contract length increases
- Month-to-month contracts show the **highest churn**
- Long-term contracts show the **lowest churn**

**Implication:**  
Contract type is a **strong and reliable predictor** of churn.

---

### 6. Number of Services vs Churn
- Churn rate drops sharply from **0 → 1 service**
- Churn increases for **low–mid service counts**
- Churn decreases significantly for **high service counts (≥3)**
- Customers with many services (7–8) show **very low churn**

**Interpretation:**  
Higher service adoption is associated with stronger customer lock-in.

**Implication:**  
Number of services is a useful **aggregate engagement feature**.

---

### 7. Individual Service Effects
- Services contribute **unequally** to churn prediction
- Support and security services show stronger associations than entertainment services

**Implication:**  
Service-specific features should be weighted differently during modeling.

---

### 8. Partner & Dependents
- Customers with partners or dependents show **lower churn**
- Relationship strength is **weak to moderate**

**Implication:**  
These features provide contextual stability signals but are secondary predictors.

---

## Data Leakage Awareness

- **Tenure** and **TotalCharges** encode **survival time**
- They are predictive but carry **partial leakage risk**
- TotalCharges is largely redundant with tenure

**Implication:**  
These features should be handled cautiously (e.g., redundancy checks, ablation testing).

---

## Overall EDA Conclusion

Churn in this dataset is driven by a combination of:
- **Contract commitment**
- **Customer engagement (services)**
- **Lifecycle stage (tenure)**

Some features are highly predictive but encode survival information and must be treated carefully during modeling.

---

## Notes for Modeling
- Avoid relying on accuracy alone
- Monitor survivorship leakage
- Prioritize contract and engagement features
- Evaluate redundancy between tenure-based features
