# src/config.py

# --- Data Configuration ---
DATA_URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
TARGET_COLUMN = 'Churn'
CUSTOMER_ID_COLUMN = 'customerID'

# --- Model & Experiment Configuration ---
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "churn_prediction_v2"
MODEL_REGISTRY_NAME = "churn_xgboost_model"
OPTUNA_N_TRIALS = 50  # Number of hyperparameter tuning trials

# --- Feature Engineering Configuration ---
COLUMNS_TO_DROP = [CUSTOMER_ID_COLUMN]
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]