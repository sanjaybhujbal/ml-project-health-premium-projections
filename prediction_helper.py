# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib
from pathlib import Path


# ------------------------------------------------------------------
# Load models & scalers safely
# ------------------------------------------------------------------
ARTIFACTS_PATH = Path("artifacts")

try:
    model_young = joblib.load(ARTIFACTS_PATH / "model_young.joblib")
    model_rest = joblib.load(ARTIFACTS_PATH / "model_rest.joblib")
    scaler_young = joblib.load(ARTIFACTS_PATH / "scaler_young.joblib")
    scaler_rest = joblib.load(ARTIFACTS_PATH / "scaler_rest.joblib")
except FileNotFoundError as e:
    raise RuntimeError(f"❌ Model/Scaler file missing: {e}")


# ------------------------------------------------------------------
# Medical risk normalization
# ------------------------------------------------------------------
def calculate_normalized_risk(medical_history: str) -> float:
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0,
    }

    if not medical_history:
        return 0.0

    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(d, 0) for d in diseases)

    max_score = 14  # 8 (heart) + 6 (diabetes / BP)
    return total_risk_score / max_score


# ------------------------------------------------------------------
# Input preprocessing
# ------------------------------------------------------------------
def preprocess_input(input_dict: dict) -> pd.DataFrame:
    expected_columns = [
        "age",
        "number_of_dependants",
        "income_lakhs",
        "insurance_plan",
        "genetical_risk",
        "normalized_risk_score",
        "gender_Male",
        "region_Northwest",
        "region_Southeast",
        "region_Southwest",
        "marital_status_Unmarried",
        "bmi_category_Obesity",
        "bmi_category_Overweight",
        "bmi_category_Underweight",
        "smoking_status_Occasional",
        "smoking_status_Regular",
        "employment_status_Salaried",
        "employment_status_Self-Employed",
    ]

    insurance_plan_encoding = {"Bronze": 1, "Silver": 2, "Gold": 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # ---------------------------
    # Categorical encoding
    # ---------------------------
    if input_dict.get("Gender") == "Male":
        df["gender_Male"] = 1

    region = input_dict.get("Region")
    if region in ["Northwest", "Southeast", "Southwest"]:
        df[f"region_{region}"] = 1

    if input_dict.get("Marital Status") == "Unmarried":
        df["marital_status_Unmarried"] = 1

    bmi = input_dict.get("BMI Category")
    if bmi in ["Obesity", "Overweight", "Underweight"]:
        df[f"bmi_category_{bmi}"] = 1

    smoking = input_dict.get("Smoking Status")
    if smoking in ["Occasional", "Regular"]:
        df[f"smoking_status_{smoking}"] = 1

    employment = input_dict.get("Employment Status")
    if employment in ["Salaried", "Self-Employed"]:
        df[f"employment_status_{employment}"] = 1

    # ---------------------------
    # Numerical inputs
    # ---------------------------
    df["insurance_plan"] = insurance_plan_encoding.get(
        input_dict.get("Insurance Plan"), 1
    )
    df["age"] = input_dict.get("Age", 0)
    df["number_of_dependants"] = input_dict.get("Number of Dependants", 0)
    df["income_lakhs"] = input_dict.get("Income in Lakhs", 0)
    df["genetical_risk"] = input_dict.get("Genetical Risk", 0)

    # ---------------------------
    # Derived features
    # ---------------------------
    df["normalized_risk_score"] = calculate_normalized_risk(
        input_dict.get("Medical History", "")
    )

    df = handle_scaling(df["age"].iloc[0], df)

    return df


# ------------------------------------------------------------------
# Scaling logic
# ------------------------------------------------------------------
def handle_scaling(age: int, df: pd.DataFrame) -> pd.DataFrame:
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object["cols_to_scale"]
    scaler = scaler_object["scaler"]

    # Dummy column required by scaler (legacy design)
    df["income_level"] = 0

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop(columns="income_level", inplace=True)

    return df


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def predict(input_dict: dict) -> int:
    input_df = preprocess_input(input_dict)

    model = model_young if input_dict.get("Age", 0) <= 25 else model_rest
    prediction = model.predict(input_df)

    return int(prediction[0])
