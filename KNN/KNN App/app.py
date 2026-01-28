import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Customer Risk Prediction System (KNN)", layout="wide")

st.title("📊 Customer Risk Prediction System (KNN)")
st.write("This system predicts customer risk by comparing them with similar customers.")

# -----------------------------
# Load Dataset (SAFE LOAD)
# -----------------------------
try:
    data = pd.read_csv("credit_risk_dataset copy.csv")
    st.success("Dataset loaded successfully ✅")
except Exception as e:
    st.error("Dataset not found ❌")
    st.stop()

# -----------------------------
# Feature Selection
# -----------------------------
X = data[['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']]
y = data['loan_status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", 10000, 200000, 50000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", 1000, 500000, 150000, step=1000)
credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
k_value = st.sidebar.slider("K Value", 1, 15, 5)

credit_hist_length = 5 if credit_history == "Yes" else 0

# -----------------------------
# Predict Button
# -----------------------------
if st.sidebar.button("Predict Customer Risk"):

    user_data = np.array([[age, income, loan_amount, credit_hist_length]])
    user_data_scaled = scaler.transform(user_data)

    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_scaled, y)

    prediction = knn.predict(user_data_scaled)[0]
    distances, indices = knn.kneighbors(user_data_scaled)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown("<h2 style='color:red;'>🔴 High Risk Customer</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>🟢 Low Risk Customer</h2>", unsafe_allow_html=True)

    st.subheader("Nearest Neighbors Explanation")
    st.write("Number of neighbors considered:", k_value)
    st.write("Majority class among neighbors:", y.iloc[indices[0]].mode()[0])

    st.dataframe(
        data.iloc[indices[0]][
            ['person_age','person_income','loan_amnt',
             'cb_person_cred_hist_length','loan_status']
        ]
    )

    st.info(
        "This decision is based on similarity with nearby customers in feature space."
    )
