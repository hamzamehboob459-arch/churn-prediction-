import streamlit as st
import pandas as pd
import joblib

# ======================
# LOAD MODEL FILES
# ======================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("Upload CSV OR Predict single customer")


# ======================
# ENCODING FUNCTION
# ======================
def preprocess(df):

    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    df_scaled = scaler.transform(df)
    return df_scaled


# ======================
# SINGLE CUSTOMER FORM
# ======================
st.header("🔹 Single Customer Prediction")

with st.form("single_form"):

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

    processed = preprocess(row)

    pred = model.predict(processed)[0]

    if pred == 1:
        st.error("❌ Customer WILL CHURN")
    else:
        st.success("✅ Customer WILL NOT CHURN")


# ======================
# CSV UPLOAD SECTION
# ======================
st.header("📂 Bulk Prediction (CSV Upload)")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    # Preprocess
    processed = preprocess(df)

    predictions = model.predict(processed)

    df["Prediction"] = predictions

    df["Result"] = df["Prediction"].apply(
        lambda x: "WILL CHURN ❌" if x == 1 else "WILL NOT CHURN ✅"
    )

    st.success("Prediction Completed!")

    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "⬇ Download Results",
        csv,
        "churn_predictions.csv",
        "text/csv"
    )
