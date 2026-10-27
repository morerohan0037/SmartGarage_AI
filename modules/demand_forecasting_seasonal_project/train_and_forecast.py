"""Complete executable seasonal demand forecasting pipeline.
Run: python train_and_forecast.py
Uses the bundled seed-42 dataset; never deletes the original data.
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
FEATURES = ['day_of_week','day_of_month','month','quarter','day_of_year','week_of_year',
'is_weekend','year','trend_idx','dow_sin','dow_cos','doy_sin','doy_cos','doy2_sin',
'doy2_cos','lag_1','lag_7','lag_14','lag_28','lag_364','rolling_7','rolling_28','rolling_56']
LAGS=[1,7,14,28,364]

def build_features(raw):
    daily=raw.groupby('date',as_index=False)['demand'].sum().sort_values('date').reset_index(drop=True)
    d=daily.date.dt
    daily['day_of_week']=d.dayofweek
    daily['day_of_month']=d.day
    daily['month']=d.month
    daily['quarter']=d.quarter
    daily['day_of_year']=d.dayofyear
    daily['week_of_year']=d.isocalendar().week.astype(int)
    daily['is_weekend']=(daily.day_of_week>=5).astype(int)
    daily['year']=d.year
    daily['trend_idx']=np.arange(len(daily))
    for name,angle in [('dow',2*np.pi*daily.day_of_week/7),('doy',2*np.pi*daily.day_of_year/365.25),('doy2',4*np.pi*daily.day_of_year/365.25)]:
        daily[f'{name}_sin']=np.sin(angle)
        daily[f'{name}_cos']=np.cos(angle)
    for lag in LAGS: daily[f'lag_{lag}']=daily.demand.shift(lag)
    for n in (7,28,56):daily[f'rolling_{n}']=daily.demand.shift(1).rolling(n).mean()
    return daily

def features_for_future(date, history, trend_idx):
    dow=date.dayofweek; doy=date.dayofyear
    row={'day_of_week':dow,'day_of_month':date.day,'month':date.month,'quarter':date.quarter,
         'day_of_year':doy,'week_of_year':int(date.isocalendar().week),'is_weekend':int(dow>=5),
         'year':date.year,'trend_idx':trend_idx,
         'dow_sin':np.sin(2*np.pi*dow/7),'dow_cos':np.cos(2*np.pi*dow/7),
         'doy_sin':np.sin(2*np.pi*doy/365.25),'doy_cos':np.cos(2*np.pi*doy/365.25),
         'doy2_sin':np.sin(4*np.pi*doy/365.25),'doy2_cos':np.cos(4*np.pi*doy/365.25)}
    for lag in LAGS:row[f'lag_{lag}']=history.get(date-pd.Timedelta(days=lag),np.nan)
    for n in (7,28,56):
        vals=[history[date-pd.Timedelta(days=k)] for k in range(1,n+1) if date-pd.Timedelta(days=k) in history]
        row[f'rolling_{n}']=float(np.mean(vals)) if vals else np.nan
    return row

def get_models():
    models={
        'Linear Regression':LinearRegression(),
        'Ridge':Ridge(alpha=5.0),
        'Random Forest':RandomForestRegressor(n_estimators=250,max_depth=16,min_samples_leaf=2,random_state=SEED,n_jobs=-1),
        'Gradient Boosting':GradientBoostingRegressor(n_estimators=220,learning_rate=.04,max_depth=3,random_state=SEED),
        'Hist Gradient Boosting':HistGradientBoostingRegressor(max_iter=260,learning_rate=.05,max_leaf_nodes=31,random_state=SEED),
        'Neural Network':Pipeline([('scale',StandardScaler()),('mlp',MLPRegressor(hidden_layer_sizes=(64,32),max_iter=400,early_stopping=True,random_state=SEED))])}
    try:
        from xgboost import XGBRegressor
        models['XGBoost']=XGBRegressor(n_estimators=350,max_depth=5,learning_rate=.03,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=SEED,n_jobs=-1)
    except ImportError:print('XGBoost unavailable; remaining six models will run.')
    return models

def score(y,pred):
    y=np.asarray(y);pred=np.asarray(pred)
    return {'MAE':float(mean_absolute_error(y,pred)),
            'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'MAPE':float(np.mean(np.abs(y-pred)/np.clip(y,1,None))*100),
            'R2':float(r2_score(y,pred))}

def main():
    for folder in ('models','outputs','plots'): (ROOT/folder).mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'data'/'daily_demand_seed42.csv',parse_dates=['date'])
    daily=build_features(raw)
    frame=daily.dropna().reset_index(drop=True)
    frame.to_csv(ROOT/'data'/'daily_business_demand_features_seed42.csv',index=False)
    test_start=frame.date.max()-pd.Timedelta(days=119)
    val_start=test_start-pd.Timedelta(days=90)
    train=frame[frame.date<val_start]; validation=frame[(frame.date>=val_start)&(frame.date<test_start)];test=frame[frame.date>=test_start]
    print(f'Train days: {len(train)}, validation days: {len(validation)}, test days: {len(test)}')
    results=[{'Model':'Seasonal Naive (lag 7)',**score(test.demand,test.lag_7)}]
    candidates={}
    for name,model in get_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');model.fit(train[FEATURES],train.demand)
        pred=np.clip(model.predict(test[FEATURES]),0,None)
        results.append({'Model':name,**score(test.demand,pred)})
        candidates[name]=(model,pred)
        joblib.dump(model, ROOT/'models'/f"{name.lower().replace(' ','_')}.joblib")
    metrics=pd.DataFrame(results).sort_values(['MAE','RMSE']).reset_index(drop=True)
    metrics.to_csv(ROOT/'outputs'/'model_metrics.csv',index=False)
    name=metrics.iloc[0].Model
    model,pred=(None,test.lag_7.to_numpy()) if name=='Seasonal Naive (lag 7)' else candidates[name]
    test_out=test[['date','demand']].copy();test_out['predicted_demand']=np.round(pred,2)
    test_out['error']=np.round(test_out.demand-test_out.predicted_demand,2)
    test_out.to_csv(ROOT/'outputs'/'test_forecast_predictions.csv',index=False)
    history={date:float(v) for date,v in zip(daily.date,daily.demand)}
    last=daily.date.max();future_rows=[]
    for i,date in enumerate(pd.date_range(last+pd.Timedelta(days=1),periods=90,freq='D')):
        f=features_for_future(date,history,len(daily)+i)
        forecast=float(f['lag_7']) if model is None else float(model.predict(pd.DataFrame([f])[FEATURES])[0])
        forecast=max(0,forecast);history[date]=forecast
        future_rows.append({'date':date,'forecast_demand':round(forecast,2)})
    future=pd.DataFrame(future_rows)
    future['week']=future.date.dt.to_period('W').astype(str)
    future['month']=future.date.dt.to_period('M').astype(str)
    future.to_csv(ROOT/'outputs'/'future_90_day_forecast.csv',index=False)
    future.groupby('week',as_index=False).forecast_demand.sum().to_csv(ROOT/'outputs'/'future_weekly_forecast.csv',index=False)
    future.groupby('month',as_index=False).forecast_demand.sum().to_csv(ROOT/'outputs'/'future_monthly_forecast.csv',index=False)
    week=daily.groupby('day_of_week',as_index=False).demand.mean();week['day_name']=week.day_of_week.map(dict(enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])))
    week.to_csv(ROOT/'outputs'/'weekday_demand_pattern.csv',index=False)
    month=daily.groupby('month',as_index=False).demand.mean();month['month_name']=month.month.map(dict(enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],1)))
    month.to_csv(ROOT/'outputs'/'monthly_demand_pattern.csv',index=False)
    raw.groupby(['branch','service_type']).demand.agg(['mean','sum']).reset_index().to_csv(ROOT/'data'/'branch_service_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5));p=metrics.sort_values('MAE');ax.barh(p.Model,p.MAE);ax.set(xlabel='MAE (jobs/day)',title='Forecast model comparison');fig.tight_layout();fig.savefig(ROOT/'plots'/'model_comparison_mae.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));ax.plot(test_out.date,test_out.demand,label='Actual');ax.plot(test_out.date,test_out.predicted_demand,label='Predicted');ax.legend();ax.set(title=f'Test one-step forecasts: {name}',ylabel='Jobs/day');fig.tight_layout();fig.savefig(ROOT/'plots'/'test_actual_vs_predicted.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));ax.plot(daily.date.tail(240),daily.demand.tail(240),label='Historical');ax.plot(future.date,future.forecast_demand,label='Recursive 90-day forecast');ax.legend();ax.set(title='90-day demand forecast',ylabel='Jobs/day');fig.tight_layout();fig.savefig(ROOT/'plots'/'future_90_day_forecast.png',dpi=180);plt.close(fig)
    for fname,x,y,title in [('weekday_seasonality.png',week.day_name,week.demand,'Weekly seasonality'),('monthly_seasonality.png',month.month_name,month.demand,'Monthly seasonality')]:
        fig,ax=plt.subplots(figsize=(9,5));ax.bar(x,y);ax.tick_params(axis='x',rotation=25);ax.set(ylabel='Mean requests/day',title=title);fig.tight_layout();fig.savefig(ROOT/'plots'/fname,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5));ax.plot(daily.date,daily.demand.rolling(30).mean());ax.set(title='30-day moving average',ylabel='Jobs/day');fig.tight_layout();fig.savefig(ROOT/'plots'/'long_term_trend.png',dpi=180);plt.close(fig)
    for col,fn,title in [('branch','branch_demand.png','Demand by branch'),('service_type','service_type_demand.png','Demand by service type')]:
        v=raw.groupby(col).demand.sum().sort_values();fig,ax=plt.subplots(figsize=(9,5));ax.barh(v.index,v.values);ax.set(title=title,xlabel='Historical service requests');fig.tight_layout();fig.savefig(ROOT/'plots'/fn,dpi=180);plt.close(fig)
    baseline=metrics[metrics.Model=='Seasonal Naive (lag 7)'].iloc[0]
    summary={'seed':SEED,'best_model':name,'best_metrics':{k:float(metrics.iloc[0][k]) for k in ('MAE','RMSE','MAPE','R2')},'baseline_metrics':{k:float(baseline[k]) for k in ('MAE','RMSE','MAPE','R2')},'note':'Test results are one-step-ahead predictions with actual lagged demand; 90-day future forecasts are recursive.'}
    (ROOT/'outputs'/'run_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nMODEL COMPARISON\n',metrics.round(3).to_string(index=False))
    print('\nNEXT 10 FORECAST DAYS\n',future[['date','forecast_demand']].head(10).to_string(index=False))
    print('\nSaved models, outputs and all eight plots.')

if __name__=='__main__':main()
