"""Fast demo: display already-generated 90-day demand forecast and plots.
Run: python demo_future_forecast.py
For a fresh forecast, first run python train_and_forecast.py.
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
forecast=pd.read_csv(ROOT/'outputs'/'future_90_day_forecast.csv')
metrics=pd.read_csv(ROOT/'outputs'/'model_metrics.csv')
print('BEST MODEL:',metrics.iloc[0].Model)
print('\nNEXT 14 DAYS:\n',forecast[['date','forecast_demand']].head(14).to_string(index=False))
print('\nMONTHLY TOTALS:\n',pd.read_csv(ROOT/'outputs'/'future_monthly_forecast.csv').round(1).to_string(index=False))
print('\nView plots in the plots/ folder, especially future_90_day_forecast.png')
