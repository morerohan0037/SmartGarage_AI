from pathlib import Path
import json
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'outputs/run_summary.json').read_text())
name=summary['best_by_validation_pr_auc'];threshold=summary['selected_threshold']
model=joblib.load(ROOT/'models'/f'{name.lower().replace(" ","_")}.joblib')
examples=pd.read_csv(ROOT/'outputs/sample_customer_predictions.csv')
features=list(model.feature_names_in_)
probs=model.predict_proba(examples[features])[:,1]
for i,p in enumerate(probs,1):print(f'Demo customer {i}: churn risk {p:.1%}, decision {"At risk" if p>=threshold else "Lower risk"}')
print('Trained model:',name,'| Threshold:',round(threshold,3))
