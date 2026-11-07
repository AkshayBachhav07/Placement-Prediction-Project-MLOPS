import os
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
