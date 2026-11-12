
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Create FastAPI application
app = FastAPI(
    title="Claim Outcome Prediction API",
    description="API to predict insurance claim outcomes using Machine Learning",
    version="1.0"
)

# Load trained ML pipeline
model = joblib.load("models/claim_prediction_pipeline.pkl")

# Define input data structure
class ClaimInput(BaseModel):
    payer: str
    provider: str
    cpt_code: str
    diagnosis_code: str
    billed_amount: float
    patient_age: int

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Claim Outcome Prediction API is running!"
    }

# Prediction endpoint

@app.post("/predict")
def predict_claim(claim: ClaimInput):

    input_data = pd.DataFrame([{
        "payer": claim.payer,
        "provider": claim.provider,
        "cpt_code": claim.cpt_code,
        "diagnosis_code": claim.diagnosis_code,
        "billed_amount": claim.billed_amount,
        "patient_age": claim.patient_age
    }])

    # Apply the same feature engineering used during training
    input_data["amount_per_age"] = (
        input_data["billed_amount"] / input_data["patient_age"]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_outcome": prediction
    }
