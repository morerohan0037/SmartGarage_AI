"""
Quick demo using the included saved models.
Run:
    python predict_new_request.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

FEATURES = [
    "branch", "service_type", "problem_type", "urgency", "traffic_level", "weather",
    "created_hour", "day_of_week", "month", "distance_km", "technicians_on_shift",
    "open_jobs_at_request", "emergency_jobs_open", "estimated_job_duration_min",
    "parts_stock_ratio", "dispatcher_experience_years", "customer_time_window_flexibility_min",
    "previous_customer_cancellations", "jobs_per_technician", "emergency_load_ratio",
    "is_weekend", "is_peak_hour"
]

def main():
    # Select models saved by the training script. XGBoost models are preferred if present.
    reg_path = MODELS / "waiting_xgboost.joblib"
    if not reg_path.exists():
        reg_path = MODELS / "waiting_linear_regression.joblib"
    clf_path = MODELS / "delay_xgboost.joblib"
    if not clf_path.exists():
        clf_path = MODELS / "delay_logistic_regression.joblib"
    if not reg_path.exists() or not clf_path.exists():
        print("Models not found. First run: python run_delayed_request_project.py")
        return

    row = {
        "branch": "Central", "service_type": "Garage Door Repair", "problem_type": "Broken Spring",
        "urgency": "Emergency", "traffic_level": "High", "weather": "Rain",
        "created_hour": 17, "day_of_week": 1, "month": 10, "distance_km": 18.0,
        "technicians_on_shift": 7, "open_jobs_at_request": 24, "emergency_jobs_open": 5,
        "estimated_job_duration_min": 95, "parts_stock_ratio": 0.65,
        "dispatcher_experience_years": 2.0, "customer_time_window_flexibility_min": 0,
        "previous_customer_cancellations": 0
    }
    row["jobs_per_technician"] = round(row["open_jobs_at_request"] / row["technicians_on_shift"], 3)
    row["emergency_load_ratio"] = round(row["emergency_jobs_open"] / row["technicians_on_shift"], 3)
    row["is_weekend"] = int(row["day_of_week"] >= 5)
    row["is_peak_hour"] = int(row["created_hour"] in [8, 9, 10, 16, 17, 18])
    X = pd.DataFrame([row])[FEATURES]

    reg = joblib.load(reg_path)
    clf = joblib.load(clf_path)
    wait = max(0.0, float(reg.predict(X)[0]))
    delay_prob = float(clf.predict_proba(X)[0, 1])
    sla = {"Emergency": 60, "High": 120, "Normal": 180, "Low": 240}[row["urgency"]]
    print("DEMO REQUEST: Emergency broken-spring repair")
    print(f"Predicted waiting time: {wait:.1f} minutes")
    print(f"Urgency SLA: {sla} minutes")
    print(f"Delay probability: {delay_prob:.1%}")
    print(f"Predicted delay over SLA: {max(0, wait-sla):.1f} minutes")
    print("Risk flag:", "DELAY RISK" if delay_prob >= 0.5 else "LOWER DELAY RISK")

if __name__ == "__main__":
    main()
