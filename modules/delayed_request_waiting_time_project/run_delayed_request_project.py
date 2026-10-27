"""
Executable training and demonstration script:
Delayed Request Detection + Waiting-Time Prediction

Run from this project folder:
    python run_delayed_request_project.py

The script loads the included seeded synthetic dataset, performs a chronological/random
train-validation-test split, trains regression and classification models, evaluates
them, writes predictions/metrics, and regenerates key plots.

For a clean reproducible demo, requirements.txt lists the dependencies.
XGBoost is optional; other models run if it is unavailable.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = ROOT / "plots"
for folder in (MODEL_DIR, OUTPUT_DIR, PLOT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    data_path = DATA_DIR / "service_requests_seed42.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Keep this script beside the included data/ folder."
        )
    df = pd.read_csv(data_path)

    # Required targets and derived fields are part of the included generated dataset.
    required = {"actual_waiting_time_min", "is_delayed", "sla_wait_limit_min"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    feature_cols = [
        "branch", "service_type", "problem_type", "urgency", "traffic_level", "weather",
        "created_hour", "day_of_week", "month", "distance_km", "technicians_on_shift",
        "open_jobs_at_request", "emergency_jobs_open", "estimated_job_duration_min",
        "parts_stock_ratio", "dispatcher_experience_years", "customer_time_window_flexibility_min",
        "previous_customer_cancellations", "jobs_per_technician", "emergency_load_ratio",
        "is_weekend", "is_peak_hour"
    ]
    cat_cols = ["branch", "service_type", "problem_type", "urgency", "traffic_level", "weather"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["is_delayed"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1875, random_state=SEED, stratify=train_df["is_delayed"]
    )

    def make_preprocessor():
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

    # 1) Waiting-time regression
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=8, random_state=SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, max_depth=16, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=SEED
        ),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED
        )
    }
    try:
        from xgboost import XGBRegressor
        reg_models["XGBoost"] = XGBRegressor(
            n_estimators=240, max_depth=5, learning_rate=0.04, subsample=0.9,
            colsample_bytree=0.9, objective="reg:squarederror",
            random_state=SEED, n_jobs=-1
        )
    except Exception:
        print("XGBoost not available; skipping it. Install it with: pip install xgboost")

    reg_rows, reg_pipes, reg_predictions = [], {}, {}
    for name, model in reg_models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(train_df[feature_cols], train_df["actual_waiting_time_min"])
        pred = np.clip(pipe.predict(test_df[feature_cols]), 0, None)
        reg_pipes[name] = pipe
        reg_predictions[name] = pred
        reg_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(test_df["actual_waiting_time_min"], pred),
            "RMSE": rmse(test_df["actual_waiting_time_min"], pred),
            "R2": r2_score(test_df["actual_waiting_time_min"], pred)
        })
        joblib.dump(pipe, MODEL_DIR / f"waiting_{name.lower().replace(' ', '_')}.joblib")

    reg_metrics = pd.DataFrame(reg_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    reg_metrics.to_csv(OUTPUT_DIR / "waiting_time_model_metrics.csv", index=False)
    best_reg_name = reg_metrics.iloc[0]["Model"]
    best_reg = reg_pipes[best_reg_name]

    # 2) Delay classification
    clf_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=800, class_weight="balanced", random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=12, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, max_depth=14, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=SEED
        )
    }
    try:
        from xgboost import XGBClassifier
        clf_models["XGBoost"] = XGBClassifier(
            n_estimators=240, max_depth=5, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=SEED, n_jobs=-1
        )
    except Exception:
        print("XGBoost classifier not available; skipping it.")

    clf_rows, clf_pipes, thresholds = [], {}, {}
    threshold_grid = np.arange(0.20, 0.81, 0.02)
    for name, model in clf_models.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(train_df[feature_cols], train_df["is_delayed"])
        val_prob = pipe.predict_proba(val_df[feature_cols])[:, 1]
        val_f1 = [
            f1_score(val_df["is_delayed"], (val_prob >= t).astype(int), zero_division=0)
            for t in threshold_grid
        ]
        threshold = float(threshold_grid[int(np.argmax(val_f1))])
        test_prob = pipe.predict_proba(test_df[feature_cols])[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        thresholds[name] = threshold
        clf_pipes[name] = pipe
        clf_rows.append({
            "Model": name,
            "Threshold": threshold,
            "Accuracy": accuracy_score(test_df["is_delayed"], test_pred),
            "Precision": precision_score(test_df["is_delayed"], test_pred, zero_division=0),
            "Recall": recall_score(test_df["is_delayed"], test_pred, zero_division=0),
            "F1": f1_score(test_df["is_delayed"], test_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(test_df["is_delayed"], test_prob)
        })
        joblib.dump(pipe, MODEL_DIR / f"delay_{name.lower().replace(' ', '_')}.joblib")

    clf_metrics = pd.DataFrame(clf_rows).sort_values(
        ["F1", "ROC_AUC"], ascending=False
    ).reset_index(drop=True)
    clf_metrics.to_csv(OUTPUT_DIR / "delay_detection_model_metrics.csv", index=False)
    pd.DataFrame([{"Model": k, "Threshold": v} for k, v in thresholds.items()]).to_csv(
        OUTPUT_DIR / "validation_thresholds.csv", index=False
    )
    best_clf_name = clf_metrics.iloc[0]["Model"]
    best_clf = clf_pipes[best_clf_name]
    best_threshold = float(clf_metrics.iloc[0]["Threshold"])

    # 3) Predictions for the held-out test set
    test_out = test_df[[
        "request_id", "urgency", "actual_waiting_time_min", "sla_wait_limit_min", "is_delayed"
    ]].copy()
    test_out["predicted_waiting_time_min"] = np.round(
        np.clip(best_reg.predict(test_df[feature_cols]), 0, None), 1
    )
    test_out["predicted_delay_probability"] = np.round(
        best_clf.predict_proba(test_df[feature_cols])[:, 1], 4
    )
    test_out["predicted_delayed"] = (
        test_out["predicted_delay_probability"] >= best_threshold
    ).astype(int)
    test_out["predicted_delay_minutes"] = np.round(
        np.maximum(0, test_out["predicted_waiting_time_min"] - test_out["sla_wait_limit_min"]), 1
    )
    test_out.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)

    # 4) Baselines
    urgency_mean = train_df.groupby("urgency")["actual_waiting_time_min"].mean()
    baseline_wait = test_df["urgency"].map(urgency_mean).to_numpy()
    pd.DataFrame([
        {
            "Approach": "Urgency-average baseline",
            "MAE": mean_absolute_error(test_df["actual_waiting_time_min"], baseline_wait),
            "RMSE": rmse(test_df["actual_waiting_time_min"], baseline_wait),
            "R2": r2_score(test_df["actual_waiting_time_min"], baseline_wait)
        },
        {
            "Approach": f"ML ({best_reg_name})",
            "MAE": reg_metrics.iloc[0]["MAE"],
            "RMSE": reg_metrics.iloc[0]["RMSE"],
            "R2": reg_metrics.iloc[0]["R2"]
        }
    ]).to_csv(OUTPUT_DIR / "waiting_time_baseline_comparison.csv", index=False)

    rule_pred = (
        (test_df["jobs_per_technician"] > 2.0)
        | (test_df["traffic_level"].isin(["High", "Severe"]))
        | ((test_df["weather"] == "Storm") & (test_df["open_jobs_at_request"] > 15))
    ).astype(int)
    test_delay_prob = best_clf.predict_proba(test_df[feature_cols])[:, 1]
    test_delay_pred = (test_delay_prob >= best_threshold).astype(int)
    pd.DataFrame([
        {
            "Approach": "Simple delay rule",
            "Accuracy": accuracy_score(test_df["is_delayed"], rule_pred),
            "Precision": precision_score(test_df["is_delayed"], rule_pred, zero_division=0),
            "Recall": recall_score(test_df["is_delayed"], rule_pred, zero_division=0),
            "F1": f1_score(test_df["is_delayed"], rule_pred, zero_division=0)
        },
        {
            "Approach": f"ML ({best_clf_name})",
            "Accuracy": accuracy_score(test_df["is_delayed"], test_delay_pred),
            "Precision": precision_score(test_df["is_delayed"], test_delay_pred, zero_division=0),
            "Recall": recall_score(test_df["is_delayed"], test_delay_pred, zero_division=0),
            "F1": f1_score(test_df["is_delayed"], test_delay_pred, zero_division=0)
        }
    ]).to_csv(OUTPUT_DIR / "delay_baseline_comparison.csv", index=False)

    # 5) Example predictions: features are illustrative, not real company cases
    examples = pd.DataFrame([
        {
            "sample_request": "Emergency Spring",
            "branch": "Central", "service_type": "Garage Door Repair", "problem_type": "Broken Spring",
            "urgency": "Emergency", "traffic_level": "High", "weather": "Rain", "created_hour": 17,
            "day_of_week": 1, "month": 10, "distance_km": 18.0, "technicians_on_shift": 7,
            "open_jobs_at_request": 24, "emergency_jobs_open": 5, "estimated_job_duration_min": 95,
            "parts_stock_ratio": 0.65, "dispatcher_experience_years": 2.0,
            "customer_time_window_flexibility_min": 0, "previous_customer_cancellations": 0
        },
        {
            "sample_request": "Routine Maintenance",
            "branch": "North", "service_type": "Maintenance", "problem_type": "Routine Maintenance",
            "urgency": "Low", "traffic_level": "Low", "weather": "Clear", "created_hour": 11,
            "day_of_week": 2, "month": 5, "distance_km": 5.0, "technicians_on_shift": 12,
            "open_jobs_at_request": 8, "emergency_jobs_open": 1, "estimated_job_duration_min": 45,
            "parts_stock_ratio": 0.95, "dispatcher_experience_years": 8.0,
            "customer_time_window_flexibility_min": 120, "previous_customer_cancellations": 0
        },
        {
            "sample_request": "Storm Gate Failure",
            "branch": "East", "service_type": "Automatic Gate Repair", "problem_type": "Gate Motor Failure",
            "urgency": "High", "traffic_level": "Severe", "weather": "Storm", "created_hour": 18,
            "day_of_week": 5, "month": 12, "distance_km": 26.0, "technicians_on_shift": 5,
            "open_jobs_at_request": 27, "emergency_jobs_open": 6, "estimated_job_duration_min": 135,
            "parts_stock_ratio": 0.52, "dispatcher_experience_years": 1.0,
            "customer_time_window_flexibility_min": 30, "previous_customer_cancellations": 1
        },
        {
            "sample_request": "Keypad Issue",
            "branch": "South", "service_type": "Garage Door Opener", "problem_type": "Remote/Keypad Issue",
            "urgency": "Normal", "traffic_level": "Medium", "weather": "Clear", "created_hour": 13,
            "day_of_week": 3, "month": 7, "distance_km": 7.5, "technicians_on_shift": 10,
            "open_jobs_at_request": 11, "emergency_jobs_open": 2, "estimated_job_duration_min": 35,
            "parts_stock_ratio": 0.90, "dispatcher_experience_years": 5.0,
            "customer_time_window_flexibility_min": 60, "previous_customer_cancellations": 0
        }
    ])
    examples["jobs_per_technician"] = (
        examples["open_jobs_at_request"] / examples["technicians_on_shift"]
    ).round(3)
    examples["emergency_load_ratio"] = (
        examples["emergency_jobs_open"] / examples["technicians_on_shift"]
    ).round(3)
    examples["is_weekend"] = (examples["day_of_week"] >= 5).astype(int)
    examples["is_peak_hour"] = examples["created_hour"].isin([8, 9, 10, 16, 17, 18]).astype(int)
    sla = {"Emergency": 60, "High": 120, "Normal": 180, "Low": 240}
    examples["sla_wait_limit_min"] = examples["urgency"].map(sla)
    examples["predicted_waiting_time_min"] = np.round(
        np.clip(best_reg.predict(examples[feature_cols]), 0, None), 1
    )
    examples["predicted_delay_probability"] = np.round(
        best_clf.predict_proba(examples[feature_cols])[:, 1], 4
    )
    examples["predicted_delayed"] = (
        examples["predicted_delay_probability"] >= best_threshold
    ).astype(int)
    examples["predicted_delay_minutes"] = np.round(
        np.maximum(0, examples["predicted_waiting_time_min"] - examples["sla_wait_limit_min"]), 1
    )
    examples.to_csv(OUTPUT_DIR / "sample_new_request_predictions.csv", index=False)

    # 6) Plots
    plt.figure(figsize=(9, 5))
    p = reg_metrics.sort_values("MAE")
    plt.barh(p["Model"], p["MAE"])
    plt.xlabel("MAE in minutes (lower is better)")
    plt.title("Waiting-Time Model Comparison")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "waiting_time_model_comparison.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    p = clf_metrics.sort_values("F1")
    plt.barh(p["Model"], p["F1"])
    plt.xlabel("F1 score")
    plt.title("Delayed-Request Detection Model Comparison")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "delay_detection_model_comparison.png", dpi=180)
    plt.close()

    actual = test_df["actual_waiting_time_min"].to_numpy()
    pred_wait = np.clip(best_reg.predict(test_df[feature_cols]), 0, None)
    rng = np.random.default_rng(SEED)
    ix = rng.choice(len(test_df), min(1600, len(test_df)), replace=False)
    mx = max(actual[ix].max(), pred_wait[ix].max())
    plt.figure(figsize=(6.5, 6))
    plt.scatter(actual[ix], pred_wait[ix], alpha=0.25, s=12)
    plt.plot([0, mx], [0, mx], linestyle="--")
    plt.xlabel("Actual waiting time (min)")
    plt.ylabel("Predicted waiting time (min)")
    plt.title(f"Actual vs Predicted Waiting Time ({best_reg_name})")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "actual_vs_predicted_waiting_time.png", dpi=180)
    plt.close()

    qdf = df.copy()
    qdf["queue_bucket"] = pd.cut(
        qdf["jobs_per_technician"], bins=[-0.1, 0.75, 1.5, 2.25, 3.0, 10],
        labels=["<=0.75", "0.76-1.5", "1.51-2.25", "2.26-3.0", ">3.0"]
    )
    q = qdf.groupby("queue_bucket", observed=True)["actual_waiting_time_min"].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(q.index.astype(str), q.values)
    plt.xlabel("Open jobs per technician")
    plt.ylabel("Average waiting time (min)")
    plt.title("Waiting Time by Queue Pressure")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "waiting_time_by_queue_pressure.png", dpi=180)
    plt.close()

    urg_order = ["Emergency", "High", "Normal", "Low"]
    rate = df.groupby("urgency")["is_delayed"].mean().reindex(urg_order) * 100
    plt.figure(figsize=(7, 5))
    plt.bar(rate.index, rate.values)
    plt.ylabel("Delayed requests (%)")
    plt.title("Delay Rate by Urgency")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "delay_rate_by_urgency.png", dpi=180)
    plt.close()

    traffic_order = ["Low", "Medium", "High", "Severe"]
    rate = df.groupby("traffic_level")["is_delayed"].mean().reindex(traffic_order) * 100
    plt.figure(figsize=(7, 5))
    plt.bar(rate.index, rate.values)
    plt.ylabel("Delayed requests (%)")
    plt.title("Delay Rate by Traffic")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "delay_rate_by_traffic.png", dpi=180)
    plt.close()

    prob = best_clf.predict_proba(test_df[feature_cols])[:, 1]
    pred = (prob >= best_threshold).astype(int)
    cm = confusion_matrix(test_df["is_delayed"], pred)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.xticks([0, 1], ["On time", "Delayed"])
    plt.yticks([0, 1], ["On time", "Delayed"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({best_clf_name})")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Feature importance / coefficient magnitude
    try:
        prep = best_clf.named_steps["prep"]
        names = prep.get_feature_names_out()
        mdl = best_clf.named_steps["model"]
        vals = getattr(mdl, "feature_importances_", None)
        if vals is None and hasattr(mdl, "coef_"):
            vals = np.abs(np.asarray(mdl.coef_).ravel())
        if vals is not None and len(vals) == len(names):
            fi = pd.DataFrame({"feature": names, "importance": vals}).sort_values(
                "importance", ascending=False
            )
            fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
            top = fi.head(12).sort_values("importance")
            plt.figure(figsize=(9, 6))
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Importance / coefficient magnitude")
            plt.title(f"Top Delay Predictors ({best_clf_name})")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / "feature_importance.png", dpi=180)
            plt.close()
    except Exception as exc:
        print("Feature importance plot skipped:", exc)

    summary = {
        "seed": SEED,
        "dataset_rows": int(len(df)),
        "delay_rate": float(df["is_delayed"].mean()),
        "best_waiting_time_model": best_reg_name,
        "best_waiting_time_metrics": {
            "mae": float(reg_metrics.iloc[0]["MAE"]),
            "rmse": float(reg_metrics.iloc[0]["RMSE"]),
            "r2": float(reg_metrics.iloc[0]["R2"])
        },
        "best_delay_classifier": best_clf_name,
        "best_delay_metrics": {
            "accuracy": float(clf_metrics.iloc[0]["Accuracy"]),
            "precision": float(clf_metrics.iloc[0]["Precision"]),
            "recall": float(clf_metrics.iloc[0]["Recall"]),
            "f1": float(clf_metrics.iloc[0]["F1"]),
            "roc_auc": float(clf_metrics.iloc[0]["ROC_AUC"]),
            "threshold": best_threshold
        }
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== WAITING-TIME REGRESSION RESULTS ===")
    print(reg_metrics.to_string(index=False))
    print("\n=== DELAY-DETECTION CLASSIFICATION RESULTS ===")
    print(clf_metrics.to_string(index=False))
    print("\n=== SAMPLE NEW-REQUEST PREDICTIONS ===")
    print(examples[[
        "sample_request", "predicted_waiting_time_min", "sla_wait_limit_min",
        "predicted_delay_probability", "predicted_delayed", "predicted_delay_minutes"
    ]].to_string(index=False))
    print(f"\nFinished. Models saved to: {MODEL_DIR}")
    print(f"Metrics/predictions saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
