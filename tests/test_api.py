import pytest
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
