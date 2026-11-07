import numpy as np
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
