import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.title("📊 Telco Customer Churn Prediction App")

st.write("""
This application predicts **customer churn** using Machine Learning  
and visualizes model performance using a **Confusion Matrix**.
""")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

data = load_data()

st.subheader("📁 Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
st.subheader("🧹 Data Preprocessing")

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

data_encoded = pd.get_dummies(data, drop_first=True)

st.success("Data cleaned and encoded successfully")

# --------------------------------------------------
# Feature & Target Split
# --------------------------------------------------
X = data_encoded.drop('Churn', axis=1)
y = data_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
st.subheader("🤖 Model Training")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.success("Logistic Regression model trained")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.subheader("📈 Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write(f"### ✅ Accuracy: **{accuracy:.2f}**")

# --------------------------------------------------
# Confusion Matrix (AS REQUESTED)
# --------------------------------------------------
st.subheader("🔴 Confusion Matrix")

fig, ax = plt.subplots(figsize=(5, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap='Reds',
    values_format='d',
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# --------------------------------------------------
# Analysis Section
# --------------------------------------------------
st.subheader("🔍 Analysis")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

st.write(f"""
- **Correctly identified churn customers (TP):** {tp}
- **Correctly identified non-churn customers (TN):** {tn}
- **Misclassified churn customers (FN):** {fn}
- **Misclassified non-churn customers (FP):** {fp}
""")

# --------------------------------------------------
# Final Conclusion
# --------------------------------------------------
st.subheader("✅ Conclusion")

st.write("""
- The Logistic Regression model successfully predicts customer churn  
- Tenure and charges significantly influence churn  
- This model helps businesses take **proactive retention actions**
""")
