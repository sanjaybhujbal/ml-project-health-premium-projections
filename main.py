# codebasics ML course: codebasics.io, all rights reserved

import streamlit as st
from prediction_helper import predict


# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    layout="wide",
)

st.title("🏥 Health Insurance Cost Predictor")
st.caption("Estimate your annual health insurance cost using ML")


# ------------------------------------------------------------------
# UI Options (SAFE for model)
# ------------------------------------------------------------------
categorical_options = {
    "Gender": ["Male", "Female"],
    "Marital Status": ["Unmarried", "Married"],
    "BMI Category": ["Normal", "Obesity", "Overweight", "Underweight"],
    "Smoking Status": ["No Smoking", "Occasional", "Regular"],
    "Employment Status": ["Salaried", "Self-Employed"],
    "Region": ["Northwest", "Southeast", "Southwest"],
    "Medical History": [
        "No Disease",
        "Diabetes",
        "High blood pressure",
        "Diabetes & High blood pressure",
        "Thyroid",
        "Heart disease",
        "High blood pressure & Heart disease",
        "Diabetes & Thyroid",
        "Diabetes & Heart disease",
    ],
    "Insurance Plan": ["Bronze", "Silver", "Gold"],
}


# ------------------------------------------------------------------
# Input Form (prevents unnecessary reruns)
# ------------------------------------------------------------------
with st.form("insurance_form"):
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)

    with row1[0]:
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
    with row1[1]:
        number_of_dependants = st.number_input(
            "Number of Dependants", min_value=0, max_value=20, step=1
        )
    with row1[2]:
        income_lakhs = st.number_input(
            "Income in Lakhs", min_value=0, max_value=200, step=1
        )

    with row2[0]:
        genetical_risk = st.number_input(
            "Genetical Risk (0–5)", min_value=0, max_value=5, step=1
        )
    with row2[1]:
        insurance_plan = st.selectbox(
            "Insurance Plan", categorical_options["Insurance Plan"]
        )
    with row2[2]:
        employment_status = st.selectbox(
            "Employment Status", categorical_options["Employment Status"]
        )

    with row3[0]:
        gender = st.selectbox("Gender", categorical_options["Gender"])
    with row3[1]:
        marital_status = st.selectbox(
            "Marital Status", categorical_options["Marital Status"]
        )
    with row3[2]:
        bmi_category = st.selectbox(
            "BMI Category", categorical_options["BMI Category"]
        )

    with row4[0]:
        smoking_status = st.selectbox(
            "Smoking Status", categorical_options["Smoking Status"]
        )
    with row4[1]:
        region = st.selectbox("Region", categorical_options["Region"])
    with row4[2]:
        medical_history = st.selectbox(
            "Medical History", categorical_options["Medical History"]
        )

    submit = st.form_submit_button("🔮 Predict")


# ------------------------------------------------------------------
# Input mapping → Model-safe values
# ------------------------------------------------------------------
if submit:
    # Map UI-only values to model expectations
    smoking_status_model = (
        smoking_status if smoking_status in ["Occasional", "Regular"] else "None"
    )

    bmi_category_model = (
        bmi_category if bmi_category != "Normal" else "None"
    )

    input_dict = {
        "Age": age,
        "Number of Dependants": number_of_dependants,
        "Income in Lakhs": income_lakhs,
        "Genetical Risk": genetical_risk,
        "Insurance Plan": insurance_plan,
        "Employment Status": employment_status,
        "Gender": gender,
        "Marital Status": marital_status,
        "BMI Category": bmi_category_model,
        "Smoking Status": smoking_status_model,
        "Region": region,
        "Medical History": medical_history,
    }

    try:
        prediction = predict(input_dict)
        st.success(f"💰 Predicted Health Insurance Cost: **₹ {prediction:,}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
