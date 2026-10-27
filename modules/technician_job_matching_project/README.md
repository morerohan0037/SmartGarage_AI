# Technician-Job Matching / Suitability Score

## Goal
Rank candidate technicians for each garage-door or automatic-gate service job and recommend the best match.

## Reproducible synthetic dataset
Seed: **42**

- 2,500 service jobs
- 40 technicians
- 8 candidate technicians per job
- 20,000 technician-job candidate rows

The target `true_suitability_score` ranges from 0 to 100. It depends on realistic factors including technician-job skill fit, parts availability, branch match, experience, historical first-time-fix rate, callback history, workload, distance, availability, urgency interactions, and controlled random noise.

## Models trained
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- Neural Network (MLP) Regressor

## Best model
**Neural Network**
- MAE: 4.284
- RMSE: 5.402
- R²: 0.975

## Ranking quality
- Exact Top-1 agreement with hidden oracle: 72.4%
- Hidden oracle technician captured within ML Top-3: 97.8%

## Operational comparison
Nearest/Available baseline:
- Suitability: 48.75
- First-time fix: 63.4%
- SLA met: 93.4%
- Response: 67.0 min

Weighted-rule baseline:
- Suitability: 76.59
- First-time fix: 83.6%
- SLA met: 50.8%
- Response: 152.9 min

ML ranking:
- Suitability: 80.77
- First-time fix: 86.4%
- SLA met: 63.2%
- Response: 132.0 min

## Interpretation
The nearest technician is not necessarily the best technician. The ML ranker learns a multi-factor suitability score that considers skill, inventory, historical performance, workload, travel, availability, and job characteristics simultaneously.

## Academic note
This is a reproducible synthetic field-service prototype. It is not real Precision Garage Door Service operational data.
