# Customer Churn & Retention Prediction

This project uses **the exact user-uploaded CSV** (`data/customer_churn_business_dataset.csv`). No synthetic customer rows are generated. Random seed `42` controls the reproducible splits and models.

## Run locally

From the extracted project folder:

```bash
python -m pip install -r requirements.txt
python run_churn_project.py
python predict_new_customer.py
```

Run All in `customer_churn_retention_demo.ipynb` for an interactive summary of the saved outputs.

## Targets and ML pipeline

- Target: `churn` (1 = churned, 0 = retained).
- Dataset: 10,000 rows; 32 columns.
- 60% train, 20% validation, 20% test with seeded stratification.
- Categorical imputation + one-hot encoding; numeric median imputation + scaling, fit on **training rows only**.
- Exclude identity and sensitive demographic fields (`customer_id`, `gender`, `age`, `country`, `city`) from predictors.
- Use validation PR-AUC to select best model and validation F1 to select its classification threshold.
- Test set is reserved for reporting metrics, predictions, charts and lift.

## Models
Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Hist Gradient Boosting, MLP Neural Network, XGBoost (if available). CatBoost was omitted because the installed version is incompatible with the installed scikit-learn pipeline API.

## Run outcomes
- Best model by validation PR-AUC: **Gradient Boosting**
- Test ROC-AUC: **0.8162**; test PR-AUC: **0.3372**.
- Accuracy **79.8%**, Precision **27.7%**, Recall **60.8%**, F1 **0.380** at validation-selected threshold.
- 10.2% churn rate in test set. Majority-class baseline achieves 89.8% accuracy but identifies no churners.
- Top 20% highest-risk customers have 28.5% observed churn, accounting for 55.9% of churners (2.79x lift).

## Contents
- `data/`: original supplied dataset.
- `run_churn_project.py`: end-to-end executable data preparation, training, evaluation, reporting and plotting.
- `predict_new_customer.py`: demo predictions from unseen test examples (their labels are not used for scoring).
- `models/`: seven trained full preprocessing+classifier pipelines.
- `outputs/`: metrics, threshold selection, actual vs predicted, prioritization lift, feature importance, sample predictions, summary.
- `plots/`: model comparison, ROC, PR, confusion matrix, importance, risk distribution, retention lift.
- `customer_churn_retention_demo.ipynb`: runnable results-inspection notebook.

## Important domain limitations
The supplied columns (logins, sessions, software features, monthly subscription contracts) describe a **subscription/digital-service setting**, not garage-door or automatic-gate customers. This is a churn-modeling reference that needs garage-service event fields and retraining before business deployment. The source dataset's provenance has not been verified, so don't describe it as a real company dataset without evidence.

There is no documented prediction snapshot date or outcome horizon. Variables such as survey responses or customer satisfaction must be confirmed to predate the eventual churn outcome before making real prospective claims. Model correlation does not prove retention interventions will work. Financial lift is not estimated without real intervention/cost data.
