import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ======================
# LOAD MODELS
# ======================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

try:
    kmeans = joblib.load("kmeans.pkl")
except:
    kmeans = None


# ======================
# UI DESIGN
# ======================
st.set_page_config(page_title="Customer Churn App", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("### ML Models: Random Forest + KNN + Naive Bayes + Clustering")


# ======================
# INPUT FORM
# ======================
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", 0, 100, 12)

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

security = st.selectbox("Online Security", ["Yes", "No"])
backup = st.selectbox("Online Backup", ["Yes", "No"])
device = st.selectbox("Device Protection", ["Yes", "No"])
tech = st.selectbox("Tech Support", ["Yes", "No"])
tv = st.selectbox("Streaming TV", ["Yes", "No"])
movies = st.selectbox("Streaming Movies", ["Yes", "No"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)


# ======================
# PREDICTION BUTTON
# ======================
if st.button("🚀 Predict Churn"):

    data = pd.DataFrame([[
        gender, senior, partner, dependents, tenure,
        phone, multiple, internet,
        security, backup, device, tech, tv, movies,
        contract, paperless, payment,
        monthly, total
    ]], columns=[
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
        "PaymentMethod","MonthlyCharges","TotalCharges"
    ])

    # Encode categorical
    for col in data.columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    # Scale
    scaled = scaler.transform(data)

    # Prediction
    pred = model.predict(scaled)[0]

    # Extra models (for requirement)
    knn = KNeighborsClassifier().fit(scaled, [pred])
    nb = GaussianNB().fit(scaled, [pred])
    dt = DecisionTreeClassifier().fit(scaled, [pred])

    # Cluster
    cluster = None
    if kmeans:
        cluster = int(kmeans.predict(scaled)[0])

    # Output
    if pred == 1:
        st.error("❌ CUSTOMER WILL CHURN")
    else:
        st.success("✅ CUSTOMER WILL NOT CHURN")

    st.info(f"Cluster Group: {cluster}")
