!pip install gradio -q

import gradio as gr
import numpy as np


best_model = models[best_model]
scaler = scaler


def predict_churn(
    gender, senior, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges
):


    def encode(val):
        return 1 if val == "Yes" else 0

    input_data = np.array([[
        encode(gender),
        senior,
        encode(partner),
        encode(dependents),
        tenure,
        encode(phone_service),
        encode(multiple_lines),
        encode(internet_service),
        encode(online_security),
        encode(online_backup),
        encode(device_protection),
        encode(tech_support),
        encode(streaming_tv),
        encode(streaming_movies),
        encode(contract),
        encode(paperless_billing),
        encode(payment_method),
        monthly_charges,
        total_charges
    ]])


    input_scaled = scaler.transform(input_data)


    pred = best_model.predict(input_scaled)[0]

    if pred == 1:
        return "❌ Customer WILL CHURN"
    else:
        return "✅ Customer WILL NOT CHURN"


demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Senior Citizen (0/1)"),
        gr.Radio(["Yes", "No"], label="Partner"),
        gr.Radio(["Yes", "No"], label="Dependents"),
        gr.Number(label="Tenure (months)"),
        gr.Radio(["Yes", "No"], label="Phone Service"),
        gr.Radio(["Yes", "No"], label="Multiple Lines"),
        gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Radio(["Yes", "No"], label="Online Security"),
        gr.Radio(["Yes", "No"], label="Online Backup"),
        gr.Radio(["Yes", "No"], label="Device Protection"),
        gr.Radio(["Yes", "No"], label="Tech Support"),
        gr.Radio(["Yes", "No"], label="Streaming TV"),
        gr.Radio(["Yes", "No"], label="Streaming Movies"),
        gr.Radio(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Radio(["Yes", "No"], label="Paperless Billing"),
        gr.Radio(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs="text",
    title="📊 Customer Churn Prediction System",
    description="Enter customer details and predict whether they will leave or stay."
)

demo.launch()
