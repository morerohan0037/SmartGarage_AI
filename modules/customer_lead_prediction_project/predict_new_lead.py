"""Prediction-only demo after training. Run: python predict_new_lead.py"""
from pathlib import Path
import json,joblib,pandas as pd
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'outputs/run_summary.json').read_text())
name=summary['best_model_by_pr_auc'].lower().replace(' ','_')
model=joblib.load(ROOT/'models'/f'{name}.joblib')
lead={'branch':'Central','service_type':'New Garage Door','lead_source':'Referral','property_type':'Single Family',
      'customer_type':'Residential','urgency':'High','contact_channel':'Phone','quote_requested':1,
      'door_age_years':18,'previous_repairs':3,'prior_customer':1,'response_time_minutes':12,
      'quote_amount_usd':2200,'budget_usd':2600,'distance_km':7,'competitor_quotes':1,
      'warranty_interest':1,'financing_interest':0,'appointment_slots_next_3days':4,
      'sales_rep_experience_years':7,'month':10,'day_of_week':2,'is_weekend':0,
      'price_to_budget_ratio':round(2200/2600,3),'available_slot_indicator':1}
p=float(model.predict_proba(pd.DataFrame([lead]))[0,1]);th=summary['threshold_selected_on_validation']
print('SAMPLE: Referral for a new garage door')
print(f'Conversion probability: {p:.1%}')
print(f'Classification: {"LIKELY TO CONVERT" if p>=th else "UNLIKELY TO CONVERT"}')
print(f'Model: {summary["best_model_by_pr_auc"]} / threshold: {th:.3f}')
print('This is an illustrative prediction using synthetic-data-trained models.')
