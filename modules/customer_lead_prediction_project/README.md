# Customer Lead Prediction | Garage Door & Automatic Gate Service

## Goal
Predict whether a new lead will **convert into a paying customer** using information available at the lead/quotation stage. Rank leads for sales follow-up, and show possible improvement over a simple non-ML lead prioritization rule.

**This is a reproducible synthetic prototype, not customer data from Precision Garage Door Service or any real garage-door company.**

## Public dataset research
Public lead-scoring sources were reviewed:
- [X Education lead scoring, Hugging Face](https://huggingface.co/datasets/shawhin/lead-scoring-x): lead sources and conversion labels, but education-sector leads and post-contact activity are not the same as garage repair enquiries.
- [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing): reliable general conversion classification dataset, but bank deposits are not garage/gate service conversions. Its `duration` feature is only known after calling and must not be used for a pre-contact prediction.

The main data is therefore **synthetically generated with `numpy.random.default_rng(42)`**. Running the training script generates `data/leads_synthetic_seed42.csv` again from code.

## Quick start (Windows, macOS, Linux)
Extract the ZIP, open a terminal inside the `customer_lead_prediction_project` directory, then:

```bash
python -m pip install -r requirements.txt
python run_customer_lead_project.py
python predict_new_lead.py
```

The first script generates the seeded data, trains and evaluates all models, saves model artifacts, creates graphs, evaluates a simple rule baseline, and writes model predictions. The second script loads the saved best model and scores a new example lead **without retraining**.

For a notebook demo, open `customer_lead_prediction_demo.ipynb` in Jupyter and choose **Run All**. Its first executable cell can regenerate everything using the training script.

## Data
- 16,000 synthetic leads
- Seed 42
- Dates from January 2024 onwards (synthetic dates)
- Target: `converted` (0 = no purchase, 1 = purchase)
- Examples: service type, source, priority, quoted price, stated budget, equipment age, past repairs, customer status, response delay, staff availability, technician/company proximity, sales experience
- **No labels or post-sale outcomes in predictor variables.** The target and IDs are excluded.

**Splitting:** 65% chronological training (10,400), 15% validation (2,400), 20% test (3,200). Decision thresholds are selected on the validation set only. Test results are held out until evaluation. The score is a conditional *synthetic* conversion estimate, not a validated real-world probability.

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Hist Gradient Boosting
5. Neural Network (MLP)
6. XGBoost

The best model is selected on **test PR-AUC for this educational demonstration**. In a rigorous real-world deployment, select the model on validation data too, and use the test period for one final unbiased comparison.

## Seeded run results
- Best model by PR-AUC: **Logistic Regression**
- ROC-AUC: **0.671**
- PR-AUC (average precision): **0.343**
- F1 at validation-selected threshold: **0.408**
- Overall test conversion rate: **20.69%**
- Top-scoring 20% conversion rate: **35.78%**
- Lift at 20%: **1.73×**

Read `outputs/model_metrics.csv` and `outputs/baseline_comparison.csv` for comparisons. This is a moderate signal, deliberately not a perfect, unrealistic one.

## Business interpretation
By ranking incoming leads, the business can direct timely follow-ups to leads that appear most likely to buy while still keeping a fair process for lower-ranked leads. Top-decile conversion and lift are more useful than accuracy alone for scarce sales resources.

The table `outputs/targeting_business_scenario.csv` includes a **hypothetical** calculation at $20 outreach cost and $160 gross contribution per conversion. These figures are illustrative and are not observed company revenue. The ranking evaluation shows correlation, **not a causal effect** of contacting leads; proper incremental impact requires a randomized outreach test.

## Folder structure
```
customer_lead_prediction_project/
  run_customer_lead_project.py     # Full reproducible pipeline (seeded generator + ML)
  predict_new_lead.py             # Quick saved-model prediction
  customer_lead_prediction_demo.ipynb
  requirements.txt
  README.md
  run_log.txt                    # Output from execution
  data/
    leads_synthetic_seed42.csv
    data_dictionary.csv
  models/                         # Six serialized ML pipelines
  outputs/
    model_metrics.csv
    baseline_comparison.csv
    test_lead_predictions.csv
    sample_new_lead_predictions.csv
    lift_by_decile.csv
    feature_importance.csv
    targeting_business_scenario.csv
    run_summary.json
  plots/
    model_comparison_prauc.png
    roc_curves.png
    precision_recall_curves.png
    confusion_matrix.png
    lift_by_decile.png
    conversion_by_source.png
    feature_importance.png
```

## Academic limitations
1. Labels come from a hypothesized process, not customer transactions.
2. The model learns relationships imposed on the generator; impressive synthetic results would not demonstrate actual business improvements.
3. Price, budget, response time, and source must all exist at the moment scoring occurs; otherwise retrain using only truly available fields.
4. Scores do not establish the causal effect of follow-up intervention.
5. Deployment requires real consented historical lead and closed-sale data, calibration, monitoring and a controlled A/B test.
