import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction System")

# =========================
# SAFE LOAD FUNCTION
# =========================
def load_model(file):
    if os.path.exists(file):
        return joblib.load(file)
    else:
        return None


model = load_model("best_model.pkl")
scaler = load_model("scaler.pkl")
encoders = load_model("encoders.pkl")


# =========================
# IF FILES NOT FOUND
# =========================
if model is None or scaler is None or encoders is None:
    st.error("❌ Model files missing in deployment!")
    st.info("👉 Please upload: best_model.pkl, scaler.pkl, encoders.pkl to GitHub repo")
    st.stop()


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(df):

    df = df.copy()

    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    return scaler.transform(df)


# =========================
# SINGLE PREDICTION
# =========================
st.header("🔹 Single Customer Prediction")

with st.form("form"):

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

    data = pd.DataFrame([[
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

    X = preprocess(data)

    pred = model.predict(X)[0]

    if pred == 1:
        st.error("❌ CUSTOMER WILL CHURN")
    else:
        st.success("✅ CUSTOMER WILL NOT CHURN")


# =========================
# CSV UPLOAD
# =========================
st.header("📂 Bulk Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.write("Preview:")
    st.dataframe(df.head())

    X = preprocess(df)

    preds = model.predict(X)

    df["Prediction"] = preds

    df["Result"] = df["Prediction"].apply(
        lambda x: "WILL CHURN ❌" if x == 1 else "WILL NOT CHURN ✅"
    )

    st.success("Done!")

    st.dataframe(df)

    st.download_button(
        "⬇ Download Results",
        df.to_csv(index=False),
        "results.csv",
        "text/csv"
    )
