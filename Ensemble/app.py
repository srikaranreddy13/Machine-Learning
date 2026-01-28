import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

# -------------------------------
# TITLE & DESCRIPTION
# -------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.markdown("""
This system uses a **Stacking Ensemble Machine Learning model**
to predict whether a loan will be approved by combining multiple ML models
for better decision making.
""")

# -------------------------------
# LOAD & TRAIN MODELS
# -------------------------------
@st.cache_data
def train_models():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    # ✅ USE ONLY UI FEATURES
    features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Self_Employed",
        "Property_Area"
    ]

    target = "Loan_Status"

    df = df[features + [target]]

    # Handle missing values
    df[features] = SimpleImputer(strategy="median").fit_transform(df[features])

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base models
    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    lr.fit(X_train_scaled, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Stacking (meta model)
    meta_X_train = np.column_stack([
        lr.predict(X_train_scaled),
        dt.predict(X_train),
        rf.predict(X_train)
    ])

    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_train)

    return lr, dt, rf, meta_model, scaler

lr, dt, rf, meta_model, scaler = train_models()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])

credit_history = 1 if credit_history == "Yes" else 0
employment = 1 if employment == "Self-Employed" else 0
property_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
property_area = property_map[property_area]

input_data = np.array([[
    app_income, co_income, loan_amount,
    loan_term, credit_history,
    employment, property_area
]])

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------
st.subheader("🧠 Stacking Model Architecture")
st.info("""
**Base Models**
- Logistic Regression
- Decision Tree
- Random Forest

**Meta Model**
- Logistic Regression
""")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_scaled = scaler.transform(input_data)

    lr_pred = lr.predict(input_scaled)[0]
    dt_pred = dt.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]

    meta_input = np.array([[lr_pred, dt_pred, rf_pred]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input).max() * 100

    st.subheader("📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.subheader("🧠 Final Decision")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"📈 Confidence Score: **{confidence:.2f}%**")

    st.subheader("📘 Business Explanation")
    if final_pred == 1:
        st.write("""
        Based on income, credit history, and combined predictions from multiple models,
        the applicant is likely to repay the loan.  
        Therefore, the stacking model predicts **loan approval**.
        """)
    else:
        st.write("""
        Based on income, credit history, and combined predictions from multiple models,
        the applicant is unlikely to repay the loan.  
        Therefore, the stacking model predicts **loan rejection**.
        """)
