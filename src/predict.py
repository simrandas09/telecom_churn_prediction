# src/predict.py
import mlflow
import pandas as pd
import shap
import joblib
from .config import MODEL_REGISTRY_NAME

# --- Load Model and Explainer from MLflow ---
# Note: This assumes you have run the training script and promoted a model to "Production"
MODEL_PIPELINE = None
SHAP_EXPLAINER = None

def load_artifacts():
    """Loads the model pipeline and SHAP explainer from MLflow."""
    global MODEL_PIPELINE, SHAP_EXPLAINER
    
    if MODEL_PIPELINE is None:
        try:
            # Load the model from the "Production" stage
            logged_model_uri = f"models:/{MODEL_REGISTRY_NAME}/production"
            MODEL_PIPELINE = mlflow.sklearn.load_model(logged_model_uri)
            print("Model pipeline loaded successfully from MLflow Registry.")
            
            # To load the SHAP explainer, we need to find the run it was saved in
            client = mlflow.tracking.MlflowClient()
            model_version_details = client.get_latest_versions(name=MODEL_REGISTRY_NAME, stages=["Production"])[0]
            run_id = model_version_details.run_id
            
            # Download the SHAP explainer artifact
            local_path = client.download_artifacts(run_id, "shap", ".")
            SHAP_EXPLAINER = joblib.load(f"{local_path}/shap_explainer.joblib")
            print("SHAP explainer loaded successfully.")

        except Exception as e:
            print(f"Error loading artifacts from MLflow: {e}")
            # In a real app, you might have fallback logic here
            raise RuntimeError("Could not load model or explainer.")

def predict_with_explanation(data: pd.DataFrame):
    """Generates a prediction and SHAP explanation for a single instance."""
    if not MODEL_PIPELINE or not SHAP_EXPLAINER:
        load_artifacts()
        
    # Preprocess data using the pipeline's preprocessor
    preprocessor = MODEL_PIPELINE.steps[0][1]
    data_transformed = preprocessor.transform(data)
    
    # Get prediction and probability
    prediction = MODEL_PIPELINE.steps[1][1].predict(data_transformed)[0]
    probability = MODEL_PIPELINE.steps[1][1].predict_proba(data_transformed)[:, 1][0]
    
    # Get SHAP explanation
    shap_values = SHAP_EXPLAINER.shap_values(data_transformed)
    
    # Map feature names to SHAP values for a clear explanation
    feature_names = preprocessor.get_feature_names_out()
    explanation = dict(zip(feature_names, shap_values[0]))
    
    # Sort explanation by the absolute impact of each feature
    sorted_explanation = {k: v for k, v in sorted(explanation.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    return {
        "prediction": int(prediction),
        "probability_churn": float(probability),
        "explanation": sorted_explanation
    }