# Delayed Request Detection & Waiting-Time Prediction

## Goal
Predict how long a garage-door or automatic-gate customer will wait for service and detect which requests are likely to exceed their urgency-specific service SLA.

## Reproducibility
- Seed: 42
- Synthetic requests: 18,000
- Train/validation/test split: 65% / 15% / 20%

## SLA thresholds
- Emergency: 60 min
- High: 120 min
- Normal: 180 min
- Low: 240 min

Overall delay rate: 42.4%

## Waiting-time regression
Models:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Neural Network

Best model: **XGBoost**
- MAE: 14.47 min
- RMSE: 18.20 min
- R²: 0.974

## Delayed-request classification
Models:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Neural Network

Best model: **XGBoost**
- Accuracy: 93.8%
- Precision: 91.9%
- Recall: 93.4%
- F1: 0.927
- ROC-AUC: 0.988
- Validation-selected threshold: 0.42

## Main synthetic relationships
Waiting time tends to increase with queue pressure, emergency backlog, distance, traffic, storms, peak hours, low parts availability, and long expected job durations. Experienced dispatching and scheduling flexibility reduce expected delay. Emergency jobs receive priority but also have the strictest SLA.

## Business use
For every incoming request:
1. predict expected waiting time,
2. calculate delay probability,
3. compare expected wait with the SLA,
4. flag high-risk requests,
5. allow dispatcher intervention before SLA violation.

Possible interventions:
- reprioritize the request,
- reassign a technician,
- bring a technician from another branch,
- proactively update the customer,
- change the ETA.

## Baselines
- Urgency-average waiting time baseline versus ML regression
- Simple queue/traffic delay rule versus ML classifier

## Files
- `data/service_requests_seed42.csv`
- `data/data_dictionary.csv`
- `outputs/waiting_time_model_metrics.csv`
- `outputs/delay_detection_model_metrics.csv`
- `outputs/waiting_time_baseline_comparison.csv`
- `outputs/delay_baseline_comparison.csv`
- `outputs/test_set_predictions.csv`
- `outputs/sample_new_request_predictions.csv`
- `models/*.joblib`
- `plots/*.png`
- `delayed_request_waiting_time_demo.ipynb`

## Academic note
This is a reproducible synthetic simulation, not real Precision Garage Door Service data.


## Run the project yourself

1. Extract the ZIP.
2. Open a terminal in the extracted `delayed_request_waiting_time_project` folder.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run the complete training/evaluation/demo pipeline:

```bash
python run_delayed_request_project.py
```

This trains the regression and classification models, writes updated metrics and predictions, saves model files, and regenerates the plots.

5. Run the quick single-request prediction demo after training:

```bash
python predict_new_request.py
```

The included CSV is already generated with seed 42. The training script uses it directly, so you can demonstrate the ML workflow without generating the data again. XGBoost is optional; if unavailable, the script skips it and trains the other models.
