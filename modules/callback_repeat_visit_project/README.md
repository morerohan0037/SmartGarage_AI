# Callback / Repeat-Visit Prediction

## Business problem
Predict whether a garage-door or automatic-gate service job will require a repeat technician visit (`callback_repeat_visit = 1`) or will be fixed on the first visit (`0`). The prediction is intended to be made before dispatch or before the service visit is completed so the business can assign a more suitable technician, verify parts, or improve diagnosis for high-risk jobs.

## Reproducible synthetic dataset
Public garage-door callback datasets with technician, inventory, diagnosis, job, and repeat-visit labels are not readily available. This project therefore generates a domain-specific synthetic dataset with a fixed NumPy random seed of **42**.

Running `callback_repeat_visit_ml.py` recreates the same 20,000-row dataset and retrains all models.

### Target
- `callback_repeat_visit = 1`: repeat visit/callback required
- `callback_repeat_visit = 0`: first-time fix

### Important predictors
The generator encodes plausible operational relationships such as:
- missing/partial parts increase callback risk;
- technician skill below required skill increases risk;
- complex jobs and older equipment increase risk;
- previous repairs/callbacks increase risk;
- stronger remote diagnosis and technician experience reduce risk;
- some nonlinear interactions are included, such as missing parts combined with skill gaps.

The generated target still contains random noise, so outcomes are not deterministic.

## Data split
- Training: 70% (14,000 jobs)
- Validation: 15% (3,000 jobs)
- Test: 15% (3,000 jobs)
- Stratified by callback class

Threshold selection is performed on the validation set only. The test set remains untouched until final evaluation.

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. CatBoost
6. Neural Network (MLP)

## Metrics
Because callbacks are the minority class, accuracy alone is insufficient. The project reports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix counts

## Current seeded run results
| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.62 | 0.825 | 0.455 | 0.670 | 0.541 | 0.852 | 0.584 |
| CatBoost | 0.60 | 0.834 | 0.471 | 0.616 | 0.534 | 0.846 | 0.575 |
| Neural Network | 0.28 | 0.843 | 0.492 | 0.553 | 0.521 | 0.845 | 0.567 |
| Random Forest | 0.57 | 0.840 | 0.483 | 0.555 | 0.517 | 0.844 | 0.561 |
| XGBoost | 0.54 | 0.815 | 0.432 | 0.635 | 0.514 | 0.837 | 0.561 |
| Decision Tree | 0.61 | 0.801 | 0.406 | 0.620 | 0.491 | 0.815 | 0.518 |

Callback prevalence is about **15.44%**. An always-`no callback` classifier would appear to have about 84.56% accuracy but would have **0 recall for callbacks**, which is why recall/F1/ROC-AUC/PR-AUC matter.

## Example predictions
The project includes four illustrative unseen service jobs in `outputs/sample_new_job_predictions.csv`. In the current run, the deployment model flags the high-risk broken-spring job and the parts-unavailable sensor job as likely callbacks, while the simple keypad problem is low risk.

## Files
- `callback_repeat_visit_ml.py`: full generator, training, evaluation, plotting and predictions
- `callback_repeat_visit_demo.ipynb`: notebook entry point
- `predict_new_job.py`: load trained model and predict a new job
- `data/garage_callback_synthetic_seed42.csv`: generated dataset
- `data/data_dictionary.csv`: feature definitions
- `outputs/model_metrics.csv`: model comparison
- `outputs/validation_thresholds.csv`: thresholds selected on validation set
- `outputs/test_set_predictions.csv`: final test predictions
- `outputs/sample_new_job_predictions.csv`: example business predictions
- `outputs/feature_importance.csv`: deployment-model feature importance
- `plots/`: model comparison, ROC, PR, threshold and business-factor plots
- `models/`: fitted models and preprocessing object

## Run locally
```bash
pip install -r requirements.txt
python callback_repeat_visit_ml.py
python predict_new_job.py
```

## Methodological note
This dataset is **synthetic**, not real Precision Garage Door customer data. It should be described as a seeded, reproducible domain simulation created because a suitable public callback dataset was unavailable. Results demonstrate the ML workflow and business logic; they should not be presented as measured real-world company performance.
