# setup_project.py
import os
import sys

FILES = {
    # -------------------------------------------------------------
    # 1. Project Dependencies
    # -------------------------------------------------------------
    "requirements.txt": """fastapi==0.111.0
uvicorn==0.30.1
pydantic==2.7.4
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
pytest==8.2.2
httpx==0.27.0
""",

    # -------------------------------------------------------------
    # 2. Synthetic Placement Dataset Generator
    # -------------------------------------------------------------
    "src/generate_data.py": """import numpy as np
import pandas as pd
import os

def generate_placement_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)
    
    cgpa = np.round(np.random.normal(7.2, 1.1, n_samples), 2)
    cgpa = np.clip(cgpa, 5.0, 10.0)
    
    internships = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05])
    projects = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.35, 0.3, 0.15, 0.05])
    workshops = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.35, 0.15, 0.1])
    aptitude_score = np.round(np.random.normal(70, 14, n_samples), 1)
    aptitude_score = np.clip(aptitude_score, 35.0, 100.0)
    soft_skills_rating = np.round(np.random.uniform(2.5, 5.0, n_samples), 1)
    backlogs = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.75, 0.15, 0.07, 0.03])
    
    # Placement probability calculation logit
    z = (
        (cgpa - 7.0) * 1.6 +
        internships * 0.9 +
        projects * 0.4 +
        workshops * 0.25 +
        (aptitude_score - 65) * 0.06 +
        (soft_skills_rating - 3.5) * 0.8 -
        backlogs * 1.4 -
        0.5
    )
    prob = 1 / (1 + np.exp(-z))
    placement_status = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "workshops_certifications": workshops,
        "aptitude_test_score": aptitude_score,
        "soft_skills_rating": soft_skills_rating,
        "backlogs": backlogs,
        "placement_status": placement_status
    })
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/placement_data.csv", index=False)
    print(f"[INFO] Dataset successfully created at data/placement_data.csv ({n_samples} records).")

if __name__ == "__main__":
    generate_placement_dataset()
""",

    # -------------------------------------------------------------
    # 3. Model Training & Serialization Pipeline
    # -------------------------------------------------------------
    "src/train.py": """import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train():
    data_path = "data/placement_data.csv"
    if not os.path.exists(data_path):
        from src.generate_data import generate_placement_dataset
        generate_placement_dataset()
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["placement_status"])
    y = df["placement_status"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=4,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    print("=" * 55)
    print("        STUDENT PLACEMENT MODEL EVALUATION METRICS       ")
    print("=" * 55)
    print(f" Accuracy : {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall   : {rec:.4f}")
    print(f" F1-Score : {f1:.4f}")
    print(f" ROC-AUC  : {roc:.4f}")
    print("=" * 55)
    
    os.makedirs("models", exist_ok=True)
    model_output_path = "models/placement_model.joblib"
    joblib.dump(pipeline, model_output_path)
    print(f"[INFO] Model artifact successfully exported to {model_output_path}")

if __name__ == "__main__":
    train()
""",

    # -------------------------------------------------------------
    # 4. Production FastAPI Web Server
    # -------------------------------------------------------------
    "src/app.py": """import os
import joblib
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "placement_model.joblib")

app = FastAPI(
    title="Student Placement Prediction API",
    description="Production MLOps inference engine to predict student campus placement outcomes.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_artifacts():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        from src.train import train
        train()
        model = joblib.load(MODEL_PATH)

class StudentFeatures(BaseModel):
    cgpa: float = Field(..., ge=0.0, le=10.0, description="Cumulative Grade Point Average (0-10)", example=8.2)
    internships: int = Field(..., ge=0, le=10, description="Completed internships", example=2)
    projects: int = Field(..., ge=0, le=20, description="Academic/practical projects completed", example=3)
    workshops_certifications: int = Field(..., ge=0, le=15, description="Number of certified technical workshops", example=1)
    aptitude_test_score: float = Field(..., ge=0.0, le=100.0, description="Aptitude assessment score (0-100)", example=78.5)
    soft_skills_rating: float = Field(..., ge=1.0, le=5.0, description="Soft skills assessment score (1-5)", example=4.2)
    backlogs: int = Field(..., ge=0, le=20, description="Number of active academic backlogs", example=0)

class PredictionResponse(BaseModel):
    placement_status: int = Field(..., description="1 for Placed, 0 for Not Placed")
    placement_label: str = Field(..., description="'Placed' or 'Not Placed'")
    placement_probability: float = Field(..., description="Calculated probability of successful placement")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

@app.get("/", tags=["General"])
def root():
    return {
        "status": "healthy",
        "service": "Student Placement Prediction Service",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(student: StudentFeatures):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact unavailable.")
    
    df = pd.DataFrame([student.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    
    return PredictionResponse(
        placement_status=pred,
        placement_label="Placed" if pred == 1 else "Not Placed",
        placement_probability=round(prob, 4)
    )

@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(students: List[StudentFeatures]):
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact unavailable.")
    if len(students) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch records list cannot be empty.")
    
    df = pd.DataFrame([s.model_dump() for s in students])
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    
    results = [
        PredictionResponse(
            placement_status=int(p),
            placement_label="Placed" if p == 1 else "Not Placed",
            placement_probability=round(float(prob), 4)
        )
        for p, prob in zip(preds, probs)
    ]
    return BatchPredictionResponse(predictions=results)
""",

    # -------------------------------------------------------------
    # 5. Automated Test Cases (Pytest)
    # -------------------------------------------------------------
    "tests/test_api.py": """import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_placed_candidate():
    payload = {
        "cgpa": 8.8,
        "internships": 2,
        "projects": 4,
        "workshops_certifications": 2,
        "aptitude_test_score": 85.0,
        "soft_skills_rating": 4.5,
        "backlogs": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["placement_status"] in [0, 1]
    assert body["placement_label"] in ["Placed", "Not Placed"]
    assert 0.0 <= body["placement_probability"] <= 1.0

def test_predict_validation_error():
    # CGPA exceeding 10.0 threshold triggers pydantic validation rejection (422)
    invalid_payload = {
        "cgpa": 12.5,
        "internships": 1,
        "projects": 1,
        "workshops_certifications": 0,
        "aptitude_test_score": 50.0,
        "soft_skills_rating": 3.0,
        "backlogs": 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_batch_prediction():
    batch_payload = [
        {
            "cgpa": 9.1,
            "internships": 3,
            "projects": 4,
            "workshops_certifications": 2,
            "aptitude_test_score": 90.0,
            "soft_skills_rating": 4.8,
            "backlogs": 0
        },
        {
            "cgpa": 5.2,
            "internships": 0,
            "projects": 1,
            "workshops_certifications": 0,
            "aptitude_test_score": 40.0,
            "soft_skills_rating": 2.5,
            "backlogs": 3
        }
    ]
    response = client.post("/predict_batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()["predictions"]
    assert len(data) == 2
    assert data[0]["placement_probability"] > data[1]["placement_probability"]
""",

    # -------------------------------------------------------------
    # 6. Production Dockerfile
    # -------------------------------------------------------------
    "Dockerfile": """FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PORT=10000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate dataset and train the model during build if artifact not present
RUN python src/generate_data.py && python src/train.py

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
""",

    ".dockerignore": """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
.pytest_cache/
.github/
""",

    # -------------------------------------------------------------
    # 7. CI/CD Pipeline (GitHub Actions)
    # -------------------------------------------------------------
    ".github/workflows/ci-cd.yml": """name: Student Placement System CI/CD

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Source Code
        uses: actions/checkout@v4

      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Generate Training Dataset
        run: python src/generate_data.py

      - name: Train and Evaluate Machine Learning Model
        run: python src/train.py

      - name: Execute Pytest Suite
        run: pytest -v --disable-warnings tests/

      - name: Build Docker Container Image
        run: |
          docker build -t student-placement-system:latest .

      - name: Test Run Docker Container
        run: |
          docker run -d -p 10000:10000 --name test-container student-placement-system:latest
          sleep 5
          docker ps | grep test-container
          docker stop test-container
""",

    # -------------------------------------------------------------
    # 8. Render Deployment Manifest
    # -------------------------------------------------------------
    "render.yaml": """services:
  - type: web
    name: student-placement-api
    env: docker
    plan: free
    region: oregon
    branch: main
    autoDeploy: true
    envVars:
      - key: PORT
        value: 10000
""",

    ".gitignore": """__pycache__/
*.py[cod]
*$py.class
venv/
.env
.pytest_cache/
models/*.joblib
data/*.csv
.DS_Store
"""
}

def main():
    for filepath, content in FILES.items():
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[OK] Created: {filepath}")
    print("\n[SUCCESS] Entire project scaffolding generated successfully!")

if __name__ == "__main__":
    main()