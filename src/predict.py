# src/predict.py
import pandas as pd
import joblib

MODEL_PIPELINE = None

def load_model(model_path="model.pkl"):
    """Loads the model pipeline from a file."""
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        try:
            MODEL_PIPELINE = joblib.load(model_path)
            print("Model pipeline loaded successfully from model.pkl.")
        except FileNotFoundError:
            print("Error: model.pkl not found.")
            raise
    return MODEL_PIPELINE

def predict(data: pd.DataFrame):
    """Generates a prediction using the loaded model."""
    pipeline = load_model()
    
    # Get prediction and probability
    prediction = pipeline.predict(data)[0]
    probability = pipeline.predict_proba(data)[:, 1][0]
    
    return {
        "prediction": int(prediction),
        "probability_churn": float(probability)
    }