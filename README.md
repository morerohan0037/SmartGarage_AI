# SmartGarage AI: Integrated six-module Streamlit dashboard

Combines **all six original uploaded projects**. The full original data, model artifacts, scripts, outputs and evaluation plots remain under `modules/`.

## Quick start (Windows PowerShell, macOS, or Linux)

Use Python 3.11 or 3.12 where possible. Unzip the archive and open a terminal inside `SmartGarage_AI`.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install and start:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The terminal displays a local browser URL, typically http://localhost:8501.

## Pages

1. Overview: key metrics and prototype limitations.
2. Callback / Repeat Visits: live saved logistic model and fitted feature preprocessor.
3. Technician Matching: ranks actual candidate technicians from a selected synthetic job using saved neural network and fitted preprocessor.
4. Delay & Waiting Time: independent saved XGBoost classifier and regressor.
5. Demand Forecasting: saved 90-day recursive forecast from the provided trained project, model-versus-baseline comparisons. **This page displays saved model-produced forecasts** rather than re-training or claiming interactive scenario recalculation.
6. Lead Conversion: saved logistic regression classifier with editable inquiry inputs and recomputation of engineered fields.
7. Customer Churn: saved gradient boosting pipeline with editable supplied sample customer inputs.

Prediction forms start with rows from the original supplied datasets. The trained models are applied live when **Predict** is pressed. For matching, job selection uses the candidate table of the original project. For demand forecasting, original saved future predictions are displayed to preserve consistency with the original recursive forecasting process.

## Limits and presentation notes

- Models were trained separately, in most cases on independently generated **synthetic** datasets (seed 42). The churn ZIP uses the provided business dataset, without independent provenance verification.
- This is a six-model operational *dashboard*, not a claim that datasets correspond to the same real customers or shared job events.
- Cross-module predictions are not causally linked. A single end-to-end, event-driven production implementation would need consistent IDs and a common operational database.
- Some model objects were serialized under a different scikit-learn environment. Python 3.11/3.12 and installing requirements are recommended; if joblib warns about versions, retrain using each original module's script in the target environment.
- Security: saved joblib files are pickle based. Only load artifacts from sources you trust.
- No confidential customer data should be uploaded into this demo.

## Original files

All six original ZIPs were extracted into `modules/` without changing the original model artifacts or outputs. Original scripts are reference/retraining scripts and are **not** executed on dashboard launch (some retain their author's original absolute paths). Launch the integrated app via `app.py`.

## Suggested live presentation

Start on Overview; show original baselines, demonstrate an inquiry and conversion probability, select a job and rank technicians, predict delay/wait and callback risk, inspect a customer churn risk, then show 90-day demand forecasts and their seasonal-naive baseline.
