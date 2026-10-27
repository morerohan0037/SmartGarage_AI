from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'logistic_regression.joblib'
THRESHOLD = 0.62

preprocessor = joblib.load(ROOT/'models'/'preprocessor.joblib')
model = joblib.load(ROOT/'models'/MODEL_NAME)

new_job = pd.DataFrame([{
    'branch_id':'B03',
    'service_type':'Garage Door Repair',
    'problem_type':'Broken Spring',
    'door_gate_type':'Double Steel',
    'door_gate_age_years':15.0,
    'previous_repairs':3,
    'prior_callbacks':1,
    'technician_experience_years':1.2,
    'technician_skill_level':2,
    'required_skill_level':4,
    'distance_km':16.0,
    'traffic_level':'High',
    'urgency':'Emergency',
    'parts_availability':'Partial',
    'remote_diagnosis_quality':2,
    'customer_described_symptoms':1,
    'first_appointment_of_day':0,
    'weekend':1,
    'time_slot':'Evening',
    'job_complexity':4
}])

X = preprocessor.transform(new_job)
probability = float(model.predict_proba(X)[0,1])
prediction = int(probability >= THRESHOLD)
print(f'Callback probability: {probability:.3f}')
print('Prediction:', 'LIKELY CALLBACK' if prediction else 'LIKELY FIRST-TIME FIX')
