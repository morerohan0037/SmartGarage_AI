# Seasonal Demand Forecasting

## Goal
Forecast future garage-door and automatic-gate service demand so the business can plan technician staffing, inventory, branch capacity, and appointment availability.

## Reproducibility
- Seed: 42
- Historical period: 2023-01-01 to 2026-09-30
- Daily branch/service observations: 34,225
- Business-level daily observations: 1,369

## Synthetic seasonality
The seeded data includes:
- weekly seasonality,
- annual seasonality,
- stronger demand in selected winter/summer periods,
- service-type-specific seasonal patterns,
- gradual multi-year business growth,
- holiday slowdowns,
- limited campaign spikes,
- random local demand shocks.

## Models
- Seasonal Naive (same weekday previous week)
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- Hist Gradient Boosting
- XGBoost
- Neural Network (MLP)

## Features
Calendar:
- day of week
- month
- quarter
- week of year
- weekend indicator
- annual cyclic sine/cosine terms

Time-series history:
- lag 1
- lag 7
- lag 14
- lag 28
- lag 364
- rolling 7-day average
- rolling 28-day average
- rolling 56-day average
- long-term trend index

## Best seeded model
**Ridge**
- MAE: 11.24 jobs/day
- RMSE: 14.29
- MAPE: 7.12%
- R²: 0.800

## Seasonal-naive baseline
- MAE: 17.36
- RMSE: 21.85
- MAPE: 11.02%
- R²: 0.532

MAE improvement of best model versus seasonal-naive baseline: 35.2%

## Forecast outputs
- next 90 days of daily demand
- weekly forecast totals
- monthly forecast totals

See:
- `outputs/future_90_day_forecast.csv`
- `outputs/future_weekly_forecast.csv`
- `outputs/future_monthly_forecast.csv`

## Business use
The forecast can support:
- technician staffing by week,
- overtime planning,
- branch scheduling,
- expected call-center workload,
- inventory purchasing,
- emergency coverage,
- appointment slot planning.

## Main files
- `data/daily_demand_seed42.csv`
- `data/daily_business_demand_features_seed42.csv`
- `outputs/model_metrics.csv`
- `outputs/test_forecast_predictions.csv`
- `outputs/future_90_day_forecast.csv`
- `outputs/future_weekly_forecast.csv`
- `outputs/future_monthly_forecast.csv`
- `plots/*.png`
- `models/*.joblib`
- `demand_forecasting_demo.ipynb`

## Academic note
This dataset is synthetic and reproducible. It should be described as a garage-door/gate field-service demand simulation rather than actual company demand.

## Executable Python code (new)

Unzip the archive, open a terminal in the `demand_forecasting_seasonal_project` folder and run:

```bash
python -m pip install -r requirements.txt
python train_and_forecast.py
python demo_future_forecast.py
```

- `train_and_forecast.py` loads the included seeded CSV, constructs lag and seasonal features, trains all available models, compares them against the seasonal-naive baseline, saves model files and predictions, and regenerates all eight figures.
- `demo_future_forecast.py` quickly displays the generated daily and monthly forecasts without retraining.
- The original dataset, notebook, models, results and plots are retained in the ZIP.

**Evaluation note:** Test errors are for one-step-ahead forecasts using observed historical lags. The forward 90-day forecast uses recursive predicted lags and has not been independently validated as a 90-day-horizon backtest. All observations are synthetic and do not represent actual garage-service demand.
