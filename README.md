# Advanced Customer Churn Prediction API

## 1. Project Overview

This project provides an end-to-end, production-ready solution for predicting customer churn in a telecom business. Moving beyond a simple experimental notebook, this system is built with a focus on robustness, reproducibility, and deployment using modern MLOps principles.

The core of the project is a highly optimized XGBoost model that predicts the probability of a customer churning. This model is served via a high-performance REST API, containerized with Docker, making it a portable and scalable service ready for integration into larger business applications.

---

## 2. Key Features

* **Advanced Model Training:** Uses **XGBoost** for high predictive accuracy.
* **Hyperparameter Tuning:** Implements **Optuna** for efficient and intelligent optimization of model hyperparameters.
* **Experiment Tracking:** Leverages **MLflow** during the training phase to log experiments, parameters, and metrics for full reproducibility.
* **High-Performance API:** The model is served using **FastAPI**, which provides a fast, modern web framework with automatic interactive documentation (Swagger UI).
* **Containerization:** The entire application is containerized with **Docker**, ensuring a consistent and portable environment for both development and deployment.
* **Clean Architecture:** The project follows a modular structure that separates concerns (configuration, training, prediction, API), making it scalable and easy to maintain.

---

## 3. Tech Stack

* **Data Science & ML:** Pandas, Scikit-learn, XGBoost, Optuna, MLflow, SHAP
* **API & Web:** FastAPI, Uvicorn
* **MLOps & DevOps:** Docker, Git, GitHub Actions (for CI)
* **Core Language:** Python 3.9

---

## 4. Project Structure

```
telecom-churn-prediction/
├── data/                 # Contains the raw dataset (for reference)
├── notebooks/            # Jupyter notebook for initial EDA
├── src/                  # Source code for the application
│   ├── api/              # Code related to the FastAPI application
│   │   ├── main.py       # FastAPI app definition and endpoints
│   │   └── schemas.py    # Pydantic data models for API I/O
│   ├── config.py         # Centralized configuration variables
│   ├── predict.py        # Prediction logic
│   └── train.py          # Model training, tuning, and logging script
├── .github/              # GitHub-specific files
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI pipeline configuration
├── .gitignore            # Files and folders to be ignored by Git
├── Dockerfile            # Instructions for building the Docker container
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 5. How to Run This Project

This project is designed to be run with Docker. Ensure you have **Docker Desktop** installed and running.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your-username/telecom-churn-prediction.git](https://github.com/your-username/telecom-churn-prediction.git)
cd telecom-churn-prediction
```

### Step 2: Run the Training Pipeline
This command runs the training script locally. It will use MLflow to track the experiment and, most importantly, will generate the final `model.pkl` file needed for the API.
```powershell
# First, install dependencies locally for the training script
pip install -r requirements.txt

# Run the training script
python src/train.py
```

### Step 3: Build the Docker Image
This command reads the `Dockerfile` and packages your API and the trained `model.pkl` into a self-contained image.
```powershell
docker build -t churn-api-advanced .
```

### Step 4: Run the Docker Container
This command starts your API server.
```powershell
docker run -p 8000:8000 churn-api-advanced
```
Your API is now live and accessible at `http://127.0.0.1:8000`.

---

## 6. API Usage

The API is self-documenting. Once the container is running, navigate to **`http://127.0.0.1:8000/docs`** in your web browser to access the interactive Swagger UI.

From there, you can:
* Explore the `/predict` endpoint.
* Click "Try it out" to populate a sample request.
* Execute the request and see the live prediction from your model.

### Example Request (PowerShell)
```powershell
Invoke-WebRequest -Uri "[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)" -Method POST -ContentType "application/json" -Body '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

### Example Response
```json
{
  "prediction": 1,
  "probability_churn": 0.65
}
```

---
