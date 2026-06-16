import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================
# SAFE FILE LOADING
# =========================
def load_file(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"❌ Missing file: {file_name}")
        st.stop()

model = load_file("best_model.pkl")
scaler = load_file("scaler.pkl")
encoders = load_file("encoders.pkl")

st.set_page_config(page_title="Customer Churn App", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Upload CSV OR Predict single customer")


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(df):

    df = df.copy()

    # encoding
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    # scaling
    return scaler.transform(df)


# =========================
# SINGLE CUSTOMER
# =========================
st.header("🔹 Single Customer Prediction")

with st.form("single"):

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

    submit = st.form_submit_button("Predict")

if submit:

    row = pd.DataFrame([[
        gender, senior, partner, dependents, tenure,
        phone, multiple, internet,
        security, backup, device, tech,
        tv, movies,
        contract, paperless, payment,
        monthly, total
    ]], columns=[
        "gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService",
        "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
        "StreamingTV","StreamingMovies","Contract","PaperlessBilling",
        "PaymentMethod","MonthlyCharges","TotalCharges"
    ])

    X = preprocess(row)

    pred = model.predict(X)[0]

    if pred == 1:
        st.error("❌ CUSTOMER WILL CHURN")
    else:
        st.success("✅ CUSTOMER WILL NOT CHURN")


# =========================
# CSV UPLOAD
# =========================
st.header("📂 CSV Bulk Prediction")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("Preview:")
    st.dataframe(df.head())

    X = preprocess(df)

    predictions = model.predict(X)

    df["Prediction"] = predictions
    df["Result"] = df["Prediction"].apply(
        lambda x: "WILL CHURN ❌" if x == 1 else "WILL NOT CHURN ✅"
    )

    st.success("Prediction Done!")

    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Results",
        csv,
        "churn_results.csv",
        "text/csv"
    )
