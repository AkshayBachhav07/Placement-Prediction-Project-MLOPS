import os
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
