import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ABC Ltd. — Customer Retention Predictor", page_icon="📊", layout="centered")

# ---------------------------------------------------------
# Load trained models (these .pkl files come from the Colab notebook)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    log_model = joblib.load("logistic_model.pkl")
    lin_model = joblib.load("linear_model.pkl")
    scaler_clf = joblib.load("scaler_clf.pkl")
    scaler_reg = joblib.load("scaler_reg.pkl")
    clf_columns = joblib.load("clf_columns.pkl")
    reg_columns = joblib.load("reg_columns.pkl")
    return log_model, lin_model, scaler_clf, scaler_reg, clf_columns, reg_columns

log_model, lin_model, scaler_clf, scaler_reg, clf_columns, reg_columns = load_artifacts()

st.title("📊 ABC Ltd. — Customer Retention Predictor")
st.caption("A decision-support tool for account managers and team leads. Enter a customer's profile to see churn risk and expected tenure.")

st.divider()

# ---------------------------------------------------------
# Input form — plain-language, non-technical
# ---------------------------------------------------------
st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has a Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.divider()
predict_clicked = st.button("🔮 Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Build a single-row dataframe matching the training format
# ---------------------------------------------------------
def build_input_row():
    raw = {
        "Gender": 1 if gender == "Male" else 0,
        "Senior_Citizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "Phone_Service": 1 if phone == "Yes" else 0,
        "Paperless_Billing": 1 if paperless == "Yes" else 0,
        "Monthly_Charges": monthly_charges,
        # one-hots below, default 0, set True where selected
        "Multiple_Lines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "Multiple_Lines_Yes": 1 if multiple_lines == "Yes" else 0,
        "Internet_Service_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "Internet_Service_No": 1 if internet == "No" else 0,
        "Online_Security_No internet service": 1 if online_sec == "No internet service" else 0,
        "Online_Security_Yes": 1 if online_sec == "Yes" else 0,
        "Online_Backup_No internet service": 1 if online_backup == "No internet service" else 0,
        "Online_Backup_Yes": 1 if online_backup == "Yes" else 0,
        "Device_Protection_No internet service": 1 if device_prot == "No internet service" else 0,
        "Device_Protection_Yes": 1 if device_prot == "Yes" else 0,
        "Tech_Support_No internet service": 1 if tech_support == "No internet service" else 0,
        "Tech_Support_Yes": 1 if tech_support == "Yes" else 0,
        "Streaming_TV_No internet service": 1 if streaming_tv == "No internet service" else 0,
        "Streaming_TV_Yes": 1 if streaming_tv == "Yes" else 0,
        "Streaming_Movies_No internet service": 1 if streaming_movies == "No internet service" else 0,
        "Streaming_Movies_Yes": 1 if streaming_movies == "Yes" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "Payment_Method_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "Payment_Method_Electronic check": 1 if payment == "Electronic check" else 0,
        "Payment_Method_Mailed check": 1 if payment == "Mailed check" else 0,
    }
    return raw

if predict_clicked:
    raw = build_input_row()

    # --- Tenure needed as a feature for the churn model ---
    # We ask the user for current tenure separately since it's a key driver
    pass

st.divider()
st.subheader("How long has this customer been with ABC Ltd.?")
tenure_months = st.slider("Current tenure (months)", 0, 72, 12)

if predict_clicked:
    raw = build_input_row()
    raw["Tenure_Months"] = tenure_months

    # ----- Logistic regression: churn probability -----
    clf_row = pd.DataFrame([raw]).reindex(columns=clf_columns, fill_value=0)
    clf_row_scaled = scaler_clf.transform(clf_row)
    churn_proba = log_model.predict_proba(clf_row_scaled)[0, 1]
    churn_pred = "High Risk" if churn_proba >= 0.5 else "Low Risk"

    # ----- Linear regression: predicted tenure (independent of current tenure input) -----
    reg_row = pd.DataFrame([raw]).reindex(columns=reg_columns, fill_value=0)
    reg_row_scaled = scaler_reg.transform(reg_row)
    predicted_tenure = lin_model.predict(reg_row_scaled)[0]

    st.divider()
    st.subheader("📋 Prediction Results")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Churn Probability", f"{churn_proba*100:.1f}%", delta=churn_pred,
                   delta_color="inverse" if churn_pred == "High Risk" else "normal")
    with r2:
        st.metric("Predicted Customer Lifetime", f"{predicted_tenure:.0f} months")

    if churn_proba >= 0.5:
        st.error(f"⚠️ This customer has a **{churn_proba*100:.0f}% chance of churning**. Recommend proactive retention outreach.")
    else:
        st.success(f"✅ This customer has a **{churn_proba*100:.0f}% chance of churning** — currently low risk.")

    # ----- Plain-language explanation using logistic coefficients -----
    st.subheader("🔍 Why this prediction?")
    coefs = pd.Series(log_model.coef_[0], index=clf_columns)
    contributions = (clf_row_scaled[0] * coefs.values)
    contrib_series = pd.Series(contributions, index=clf_columns).sort_values(key=abs, ascending=False).head(5)

    st.caption("Top factors influencing this customer's churn risk (based on their specific profile):")
    for feat, val in contrib_series.items():
        direction = "increases" if val > 0 else "decreases"
        label = feat.replace("_", " ")
        st.write(f"- **{label}** {direction} churn risk")

    st.caption("Note: This is a decision-support estimate, not a certainty. Use alongside your own judgment and account history.")

st.divider()
st.caption("Model info: Logistic Regression (churn) — ROC-AUC ≈ 0.85 on test data. "
           "Linear Regression (tenure) — trained on ABC Ltd. historical customer data. "
           "Built for managerial decision support, not automated action.")
