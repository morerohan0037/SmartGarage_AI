"""SmartGarage AI: dashboard integrating six supplied ML projects."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MOD = ROOT / 'modules'
PROJECTS = {
 'Callback / Repeat Visits':'callback_repeat_visit_project',
 'Technician Matching':'technician_job_matching_project',
 'Delay & Waiting Time':'delayed_request_waiting_time_project',
 'Demand Forecasting':'demand_forecasting_seasonal_project',
 'Lead Conversion':'customer_lead_prediction_project',
 'Customer Churn':'customer_churn_retention_project',
}
st.set_page_config(page_title='SmartGarage AI',page_icon='🔧',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.6rem;max-width:1400px} 
[data-testid="stMetric"]{background:#f3f6fa;border:1px solid #e2e8f0;padding:14px;border-radius:12px}
.hero{background:linear-gradient(110deg,#10253e,#214b70);padding:24px 28px;border-radius:14px;color:white;margin-bottom:12px}
.hero h1{color:white;margin:0;font-size:32px}.hero p{margin:5px 0 0;color:#d9e7f8}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading trained model...')
def model_load(path):
 return joblib.load(path)
@st.cache_data(show_spinner=False)
def csv(path):
 return pd.read_csv(path)
@st.cache_data(show_spinner=False)
def summary(path):
 return json.loads(path.read_text())

def source(key):return MOD / PROJECTS[key]
def chart_box(base,name,caption=None):
 p=base/'plots'/name
 if p.exists(): st.image(str(p),caption=caption,use_container_width=True)
def plots_gallery(base):
 plots=sorted((base/'plots').glob('*.png'))
 with st.expander(f'View all {len(plots)} original evaluation plots'):
  a,b=st.columns(2)
  for i,p in enumerate(plots):
   with (a if i%2==0 else b):st.image(str(p),caption=p.stem.replace('_',' ').title(),use_container_width=True)
def table_download(df,name):
 st.download_button('Download results (CSV)',df.to_csv(index=False).encode(),file_name=name,mime='text/csv')

def sample_form(frame,features,prefix,description='Adjust features and run the saved trained model.'):
 st.caption(description+' All defaults are drawn from a supplied dataset or original demo record.')
 row=frame.iloc[0].copy()
 with st.form(prefix+'_form'):
  cols=st.columns(3)
  for i,f in enumerate(features):
   if f not in row.index:continue
   val=row[f]
   with cols[i%3]:
    if pd.api.types.is_numeric_dtype(frame[f]):
     if pd.isna(val):val=0
     if pd.api.types.is_integer_dtype(frame[f]):row[f]=st.number_input(f,value=int(val),step=1,key=prefix+f)
     else:row[f]=st.number_input(f,value=float(val),step=0.1,format='%.4f',key=prefix+f)
    else:
     choices=[str(x) for x in frame[f].dropna().unique()][:100]
     if not choices:choices=['Unknown']
     val=str(val) if pd.notna(val) else choices[0]
     if val not in choices:choices.insert(0,val)
     row[f]=st.selectbox(f,choices,index=choices.index(val),key=prefix+f)
  clicked=st.form_submit_button('🔮 Predict with saved ML model',type='primary',use_container_width=True)
 return pd.DataFrame([row])[features],clicked

def warning():st.info('Prototype outputs: generated datasets are synthetic except that the churn project uses the provided business dataset. These results are not independently verified real-world performance.')

def overview():
 st.markdown('<div class="hero"><h1>🔧 SmartGarage AI</h1><p>Six machine learning solutions connected in one service-chain operations dashboard</p></div>',unsafe_allow_html=True)
 a,b,c,d=st.columns(4)
 a.metric('Integrated ML solutions','6');b.metric('Saved models','Loaded on demand');c.metric('Evaluation','Original test sets');d.metric('Deployment','Local demonstration')
 st.subheader('Business journey')
 st.markdown('**New inquiry → Lead scoring → Technician recommendation → Wait and delay risk → Callback prevention → Retention risk**')
 st.caption('Demand forecasting runs in parallel to inform staffing and inventory. Modules are independently trained; the dashboard does not claim a unified historical customer record.')
 for label,folder in PROJECTS.items():
  s=source(label)
  with st.expander(label):
   sm=summary(s/'outputs'/'run_summary.json')
   st.json(sm,expanded=False)
 st.subheader('Comparison at a glance')
 cards=[('Callback / Repeat Visits','ROC-AUC',.8523),('Technician Matching','R²',.9751),('Delay & Waiting Time','Delay ROC-AUC',.9877),('Demand Forecasting','R²',.7996),('Lead Conversion','PR-AUC',.3425),('Customer Churn','PR-AUC',.3372)]
 st.dataframe(pd.DataFrame(cards,columns=['Module','Representative held-out metric','Reported score']),hide_index=True,use_container_width=True)
 st.caption('Metrics have different definitions and must not be ranked against each other; inspect each module for the underlying comparison.')
 warning()

def model_comparison(folder,filename='model_metrics.csv'):
 p=folder/'outputs'/filename
 if p.exists():
  st.markdown('#### Original model comparison')
  st.dataframe(csv(p),hide_index=True,use_container_width=True)

def lead():
 st.header('📞 Lead Conversion Prediction')
 s=source('Lead Conversion');sm=summary(s/'outputs/run_summary.json');m=model_load(s/'models/logistic_regression.joblib')
 df=csv(s/'data/leads_synthetic_seed42.csv');f=list(m.feature_names_in_)
 p=csv(s/'outputs/sample_new_lead_predictions.csv')
 st.caption('Predict whether an inquiry will turn into a confirmed service booking.')
 # demonstrate original first example, then interactive edit
 initial=df.iloc[[0]].copy()
 features,go=sample_form(initial if len(p)==0 else pd.concat([initial,df.head(250)]),f,'lead')
 if go:
  # calculated fields must remain consistent when dependent inputs change
  x=features.copy();x['price_to_budget_ratio']=round(float(x.iloc[0]['quote_amount_usd'])/max(1,float(x.iloc[0]['budget_usd'])),3)
  x['available_slot_indicator']=int(x.iloc[0]['appointment_slots_next_3days']>0)
  x['is_weekend']=int(x.iloc[0]['day_of_week']>=5)
  prob=float(m.predict_proba(x)[0,1]);th=sm['threshold_selected_on_validation']
  a,b=st.columns(2);a.metric('Booking conversion probability',f'{prob:.1%}');b.metric('Decision', 'Prioritize follow-up' if prob>=th else 'Lower priority')
  st.caption(f'Deployment threshold {th:.3f} selected on validation data. Synthetic training data.')
 model_comparison(s);chart_box(s,'precision_recall_curves.png');plots_gallery(s)

def churn():
 st.header('💙 Customer Churn / Retention')
 s=source('Customer Churn');sm=summary(s/'outputs/run_summary.json')
 name=sm['best_by_validation_pr_auc'].lower().replace(' ','_')
 m=model_load(s/'models'/f'{name}.joblib')
 df=csv(s/'outputs/sample_customer_predictions.csv')
 x,go=sample_form(df,list(m.feature_names_in_),'churn','Adjust a supplied demo customer record. This module uses the business dataset you uploaded.')
 if go:
  prob=float(m.predict_proba(x)[0,1]);th=sm['selected_threshold']
  a,b=st.columns(2);a.metric('Customer churn probability',f'{prob:.1%}');b.metric('Retention flag','At risk' if prob>=th else 'Lower risk')
  st.caption(f'Deployment threshold {th:.3f}; selected model: {sm["best_by_validation_pr_auc"]}.')
 model_comparison(s);chart_box(s,'risk_distribution.png');plots_gallery(s)

def callbacks():
 st.header('🔁 Callback / Repeat Visit Risk')
 s=source('Callback / Repeat Visits');sm=summary(s/'outputs/run_summary.json')
 m=model_load(s/'models/logistic_regression.joblib');prep=model_load(s/'models/preprocessor.joblib')
 df=csv(s/'data/garage_callback_synthetic_seed42.csv')
 # derive actual source columns from fitted column transformer
 features=[]
 for name,transformer,cols in prep.transformers_:
  if name!='remainder' and isinstance(cols,(list,tuple,np.ndarray)):features+=list(cols)
 features=[x for x in features if x in df.columns]
 x,go=sample_form(df.head(300),features,'callback')
 if go:
  prob=float(m.predict_proba(prep.transform(x))[0,1]);th=sm['deployment_threshold_from_validation']
  a,b=st.columns(2);a.metric('Repeat-visit probability',f'{prob:.1%}');b.metric('Callback risk','Higher risk' if prob>=th else 'Lower risk')
  st.caption(f'Threshold {th:.2f}, selected from validation data.')
 model_comparison(s);chart_box(s,'roc_curves.png');plots_gallery(s)

def delays():
 st.header('⏱️ Delayed Requests & Waiting Times')
 s=source('Delay & Waiting Time');sm=summary(s/'outputs/run_summary.json')
 clf=model_load(s/'models/delay_xgboost.joblib');reg=model_load(s/'models/waiting_xgboost.joblib')
 df=csv(s/'data/service_requests_seed42.csv');f=list(clf.feature_names_in_)
 x,go=sample_form(df.head(300),f,'delay')
 if go:
  x['jobs_per_technician']=round(float(x.iloc[0]['open_jobs_at_request'])/max(1,float(x.iloc[0]['technicians_on_shift'])),3)
  x['emergency_load_ratio']=round(float(x.iloc[0]['emergency_jobs_open'])/max(1,float(x.iloc[0]['technicians_on_shift'])),3)
  x['is_weekend']=int(x.iloc[0]['day_of_week']>=5)
  x['is_peak_hour']=int(x.iloc[0]['created_hour'] in [8,9,10,16,17,18])
  wait=max(0,float(reg.predict(x)[0]));prob=float(clf.predict_proba(x)[0,1]);th=sm['best_delay_metrics']['threshold']
  a,b,c=st.columns(3);a.metric('Predicted waiting',f'{wait:.1f} min');b.metric('Delay probability',f'{prob:.1%}');c.metric('Flag','Delay risk' if prob>=th else 'Lower risk')
  st.caption(f'Delay classification threshold {th:.2f}. The predicted wait and delay probability are outputs of two separate models.')
 st.subheader('Classification model results');model_comparison(s,'delay_detection_model_metrics.csv')
 st.subheader('Waiting-time model results');model_comparison(s,'waiting_time_model_metrics.csv')
 chart_box(s,'actual_vs_predicted_waiting_time.png');plots_gallery(s)

def matching():
 st.header('🧰 Technician–Job Matching')
 s=source('Technician Matching');sm=summary(s/'outputs/run_summary.json')
 jobs=csv(s/'data/jobs_seed42.csv');pairs=csv(s/'data/technician_job_candidates_seed42.csv')
 pre=model_load(s/'models/preprocessor.joblib');m=model_load(s/'models/neural_network.joblib')
 options=jobs['job_id'].astype(str).tolist()
 job=st.selectbox('Choose an existing service job',options[:200],index=0)
 candidates=pairs[pairs.job_id.astype(str)==job].copy()
 if candidates.empty:st.warning('No candidate technicians for this job.');return
 st.write('Selected job:',job)
 features=['job_branch','technician_branch','service_type','problem_type','urgency','required_part','equipment_age_years','previous_repairs','estimated_duration_minutes','required_skill','experience_years','relevant_skill','customer_rating','historical_first_fix_rate','callback_rate','current_workload_ratio','available_in_minutes','distance_km','travel_time_minutes','inventory_match','same_branch','skill_gap','skill_match_ratio']
 if st.button('🔮 Rank available technicians',type='primary',use_container_width=True):
  result=candidates.copy()
  result['predicted_suitability_score']=np.clip(m.predict(pre.transform(result[features])),0,100)
  result=result.sort_values('predicted_suitability_score',ascending=False)
  show=['technician_id','predicted_suitability_score','relevant_skill','distance_km','available_in_minutes','inventory_match','historical_first_fix_rate']
  st.success(f'Top recommendation: {result.iloc[0]["technician_id"]} with suitability score {result.iloc[0]["predicted_suitability_score"]:.1f}/100')
  st.dataframe(result[show].round(2),hide_index=True,use_container_width=True);table_download(result[show],f'technician_rank_{job}.csv')
 st.caption('Ranks only candidates present in the supplied synthetic candidate table. Scores are a learned proxy for suitability, not proven job outcomes.')
 model_comparison(s);st.subheader('Dispatch strategy comparison');st.dataframe(csv(s/'outputs/dispatch_strategy_kpis.csv'),hide_index=True,use_container_width=True)
 chart_box(s,'dispatch_strategy_suitability.png');plots_gallery(s)

def forecasting():
 st.header('📈 Seasonal Demand Forecasting')
 s=source('Demand Forecasting');sm=summary(s/'outputs/run_summary.json')
 forecast=csv(s/'outputs/future_90_day_forecast.csv')
 st.caption('These are the actual saved predictions from the Ridge model. They are recursive future predictions, not merely a chart illustration.')
 st.dataframe(forecast.head(12),hide_index=True,use_container_width=True)
 num=[c for c in forecast.columns if pd.api.types.is_numeric_dtype(forecast[c])]
 date=[c for c in forecast.columns if 'date' in c.lower()]
 if num:
  col=st.selectbox('Forecast output to visualize',num)
  vis=forecast.copy();vis.index=pd.to_datetime(vis[date[0]],errors='coerce') if date else np.arange(len(vis))
  st.line_chart(vis[col]);st.metric('90-day forecast total',f'{vis[col].sum():,.0f}')
 table_download(forecast,'future_90_day_forecast.csv')
 st.subheader('Original evaluation against seasonal baseline')
 a,b=st.columns(2);a.metric('Ridge test MAE',f'{sm["best_metrics"]["MAE"]:.2f}');b.metric('Seasonal baseline MAE',f'{sm["baseline_metrics"]["MAE"]:.2f}')
 model_comparison(s);chart_box(s,'future_90_day_forecast.png');plots_gallery(s)

st.sidebar.title('🔧 SmartGarage AI')
st.sidebar.caption('Integrated ML Lab Demonstration')
page=st.sidebar.radio('Navigate',['Overview']+list(PROJECTS.keys()))
st.sidebar.divider()
st.sidebar.info('All six original model projects are embedded. No retraining is required to launch the dashboard.')
try:
 if page=='Overview':overview()
 elif page=='Lead Conversion':lead()
 elif page=='Customer Churn':churn()
 elif page=='Callback / Repeat Visits':callbacks()
 elif page=='Delay & Waiting Time':delays()
 elif page=='Technician Matching':matching()
 elif page=='Demand Forecasting':forecasting()
except Exception as e:
 st.error(f'Could not run {page}: {e}')
 st.caption('Check README for troubleshooting and dependency installation.')
st.divider()
st.caption('SmartGarage AI | Educational prototype | Do not treat synthetic results as verified business performance')
