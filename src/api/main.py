# src/api/main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from .schemas import ChurnPredictionRequest, ChurnPredictionResponse
from ..predict import predict, load_model

app = FastAPI(
    title="Customer Churn Prediction API",
    description="A robust API to predict customer churn.",
    version="3.0.0"
)

@app.on_event("startup")
def startup_event():
    """Load the model on API startup."""
    load_model()

@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Prediction"])
def post_predict(request: ChurnPredictionRequest):
    try:
        data_df = pd.DataFrame([request.dict()])
        result = predict(data_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))