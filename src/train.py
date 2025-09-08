# src/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
import optuna
from sklearn.metrics import roc_auc_score
import joblib
import shap
import time

# Import configuration
from config import (
    DATA_URL, TARGET_COLUMN, COLUMNS_TO_DROP, NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MODEL_REGISTRY_NAME, OPTUNA_N_TRIALS
)

def load_and_preprocess_data(url):
    """Loads and preprocesses the Telco Churn dataset."""
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # This FutureWarning is expected and acceptable for this project.
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop(columns=COLUMNS_TO_DROP, inplace=True)
    return df

# ***** FIX IS HERE: Function now accepts feature lists as arguments *****
def get_preprocessor(numerical_features, categorical_features):
    """Returns a scikit-learn ColumnTransformer for preprocessing."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
# *********************************************************************

def objective(trial, X, y, preprocessor):
    """Optuna objective function for hyperparameter tuning."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False))])
    pipeline.fit(X_train, y_train)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    return auc

def main():
    """Main function to run the training pipeline."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_and_preprocess_data(DATA_URL)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # ***** FIX IS HERE: We now pass the imported lists into the function *****
    preprocessor = get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    # ************************************************************************

    print("--- Starting Hyperparameter Tuning with Optuna ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor), n_trials=OPTUNA_N_TRIALS)

    print(f"Best trial AUC: {study.best_value}")
    print("Best hyperparameters: ", study.best_params)

    print("\n--- Training and Logging Final Model with MLflow ---")
    with mlflow.start_run() as run:
        mlflow.log_params(study.best_params)
        final_model = xgb.XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False)
        final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', final_model)])
        final_pipeline.fit(X_train, y_train)

        y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("test_auc", auc)
        print(f"Final model Test AUC: {auc}")

        X_train_transformed = final_pipeline.named_steps['preprocessor'].fit_transform(X_train)
        explainer = shap.TreeExplainer(final_pipeline.named_steps['classifier'])
        joblib.dump(explainer, "shap_explainer.joblib")
        mlflow.log_artifact("shap_explainer.joblib", artifact_path="shap")

        joblib.dump(final_pipeline, "model.pkl")
        print("Model pipeline saved locally to model.pkl")
        
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME
        )
        print(f"Model logged and registered as '{MODEL_REGISTRY_NAME}'.")

        print("\n--- Promoting Model to Production Stage ---")
        client = MlflowClient()
        time.sleep(5)
        latest_version = client.get_latest_versions(name=MODEL_REGISTRY_NAME, stages=["None"])[0]
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=latest_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted model version {latest_version.version} to 'Production'.")

if __name__ == "__main__":
    main()