# src/api/main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from .schemas import ChurnPredictionRequest, ChurnPredictionResponse
from ..predict import predict_with_explanation, load_artifacts

app = FastAPI(
    title="Advanced Customer Churn Prediction API",
    description="An API to predict customer churn and provide model explanations using SHAP. Built with FastAPI.",
    version="2.0.0"
)

@app.on_event("startup")
def startup_event():
    """Load model and explainer artifacts on API startup."""
    try:
        load_artifacts()
    except Exception as e:
        # This will prevent the app from starting if the model can't be loaded
        raise RuntimeError(f"Startup failed: {e}")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Go to /docs for interactive documentation."}

@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Prediction"])
def predict(request: ChurnPredictionRequest):
    """
    Predicts churn for a single customer and returns the prediction,
    probability, and a SHAP-based explanation.

    **SHAP Explanation:** The 'explanation' field shows the impact of each feature
    on the prediction. Positive values push the prediction towards churn (1), while
    negative values push it away from churn.
    """
    try:
        # Convert Pydantic request to a pandas DataFrame
        data_df = pd.DataFrame([request.dict()])
        
        # Get prediction and explanation
        result = predict_with_explanation(data_df)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))