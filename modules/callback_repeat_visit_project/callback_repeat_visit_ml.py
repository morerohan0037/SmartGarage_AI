from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib

SEED = 42
N = 20000
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
PLOTS_DIR = ROOT / 'plots'
MODELS_DIR = ROOT / 'models'
OUT_DIR = ROOT / 'outputs'
for d in [DATA_DIR, PLOTS_DIR, MODELS_DIR, OUT_DIR]: d.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

# -------------------------
# 1. Synthetic data generator
# -------------------------
branches = [f'B{i:02d}' for i in range(1, 13)]
service_types = ['Garage Door Repair', 'Garage Door Opener', 'Automatic Gate Repair', 'Maintenance', 'Installation']
problem_types = [
    'Broken Spring','Cable Failure','Door Off Track','Opener/Motor Failure','Sensor Failure',
    'Roller/Hinge Issue','Remote/Keypad Issue','Gate Motor Failure','Control Board Failure',
    'Preventive Maintenance','New Installation'
]
door_types = ['Single Steel','Double Steel','Wood','Aluminum/Glass','Sectional Commercial','Swing Gate','Sliding Gate']
urgencies = ['Low','Normal','High','Emergency']
traffic_levels = ['Low','Medium','High']
parts_levels = ['Unavailable','Partial','Full']
time_slots = ['Morning','Midday','Afternoon','Evening']

service_probs = [0.38,0.18,0.20,0.14,0.10]
service_type = rng.choice(service_types, N, p=service_probs)

problem_map = {
    'Garage Door Repair': (['Broken Spring','Cable Failure','Door Off Track','Roller/Hinge Issue','Sensor Failure'], [0.34,0.18,0.17,0.18,0.13]),
    'Garage Door Opener': (['Opener/Motor Failure','Sensor Failure','Remote/Keypad Issue','Control Board Failure'], [0.42,0.22,0.20,0.16]),
    'Automatic Gate Repair': (['Gate Motor Failure','Control Board Failure','Sensor Failure','Remote/Keypad Issue'], [0.45,0.25,0.18,0.12]),
    'Maintenance': (['Preventive Maintenance','Roller/Hinge Issue','Sensor Failure'], [0.68,0.18,0.14]),
    'Installation': (['New Installation'], [1.0])
}
problem_type = []
for st in service_type:
    vals, probs = problem_map[st]
    problem_type.append(rng.choice(vals, p=probs))
problem_type = np.array(problem_type)

# Equipment age is lower for installations, otherwise right-skewed but capped.
door_age = np.where(service_type=='Installation', rng.integers(0,2,N), np.clip(rng.gamma(2.3,4.2,N),0,30)).round(1)
previous_repairs = np.clip(rng.poisson(0.12*door_age + 0.15), 0, 10)
prior_callbacks = np.minimum(previous_repairs, rng.binomial(np.maximum(previous_repairs,1), 0.16))

technician_experience = np.clip(rng.gamma(2.4,2.0,N),0.2,18).round(1)
technician_skill = np.clip(np.rint(rng.normal(3.3 + 0.09*np.minimum(technician_experience,10),0.75,N)),1,5).astype(int)

# Required skill depends on issue complexity.
complex_problem = np.isin(problem_type, ['Broken Spring','Door Off Track','Opener/Motor Failure','Gate Motor Failure','Control Board Failure','New Installation']).astype(int)
job_complexity = np.clip(np.rint(1.4 + 1.35*complex_problem + 0.045*door_age + rng.normal(0,0.65,N)),1,5).astype(int)
required_skill = np.clip(job_complexity + rng.binomial(1,0.20,N),1,5)
skill_gap = required_skill - technician_skill

distance_km = np.clip(rng.gamma(2.2,4.5,N),0.3,45).round(1)
traffic_level = rng.choice(traffic_levels,N,p=[0.32,0.46,0.22])
time_slot = rng.choice(time_slots,N,p=[0.29,0.28,0.28,0.15])
urgency = rng.choice(urgencies,N,p=[0.14,0.53,0.23,0.10])
remote_diagnosis_quality = np.clip(np.rint(rng.normal(3.3,0.95,N)),1,5).astype(int)

# Parts availability is somewhat better for common/simple issues and better diagnosis.
parts_score_latent = 0.65 + 0.18*(remote_diagnosis_quality>=4) - 0.20*(job_complexity>=4) + rng.normal(0,0.33,N)
parts_available = np.where(parts_score_latent>0.65,'Full',np.where(parts_score_latent>0.18,'Partial','Unavailable'))

first_appointment_of_day = rng.binomial(1,0.20,N)
weekend = rng.binomial(1,0.22,N)
customer_described_symptoms = rng.binomial(1, np.clip(0.56 + 0.07*(remote_diagnosis_quality-3),0.25,0.9), N)

# Service-type appropriate door/gate form.
door_type = []
for st in service_type:
    if st == 'Automatic Gate Repair':
        door_type.append(rng.choice(['Swing Gate','Sliding Gate'], p=[0.44,0.56]))
    else:
        door_type.append(rng.choice(['Single Steel','Double Steel','Wood','Aluminum/Glass','Sectional Commercial'], p=[0.24,0.31,0.16,0.12,0.17]))
door_type = np.array(door_type)

# Latent callback probability. This defines realistic relationships while preserving noise.
urg_w = pd.Series(urgency).map({'Low':-0.15,'Normal':0.0,'High':0.30,'Emergency':0.55}).to_numpy()
parts_w = pd.Series(parts_available).map({'Full':-1.05,'Partial':0.55,'Unavailable':1.65}).to_numpy()
traffic_w = pd.Series(traffic_level).map({'Low':0.0,'Medium':0.07,'High':0.16}).to_numpy()
problem_w = pd.Series(problem_type).map({
    'Broken Spring':0.35,'Cable Failure':0.20,'Door Off Track':0.38,'Opener/Motor Failure':0.44,
    'Sensor Failure':0.08,'Roller/Hinge Issue':-0.10,'Remote/Keypad Issue':-0.28,
    'Gate Motor Failure':0.52,'Control Board Failure':0.48,'Preventive Maintenance':-0.55,'New Installation':0.15
}).to_numpy()

logit = (
    -3.15
    + 0.065*door_age
    + 0.20*previous_repairs
    + 0.44*prior_callbacks
    + 0.55*np.maximum(skill_gap,0)
    - 0.10*technician_experience
    + 0.33*(job_complexity-2)
    + parts_w
    + urg_w
    + problem_w
    + traffic_w
    + 0.18*(distance_km>18)
    - 0.24*(remote_diagnosis_quality-3)
    - 0.22*customer_described_symptoms
    + 0.10*weekend
    - 0.10*first_appointment_of_day
    # Nonlinear field-service interactions that simple additive models cannot fully capture.
    + 0.95*((parts_available=='Unavailable') & (skill_gap>0))
    + 0.70*((job_complexity>=4) & (technician_experience<2.0))
    + 0.60*((door_age>15) & (previous_repairs>=3))
    + 0.55*((problem_type=='Gate Motor Failure') & (technician_skill<4))
    + 0.42*((urgency=='Emergency') & (time_slot=='Evening'))
    - 0.60*((parts_available=='Full') & (skill_gap<=0) & (remote_diagnosis_quality>=4))
    + rng.normal(0,0.58,N)
)
prob_callback = 1/(1+np.exp(-logit))
callback = rng.binomial(1, prob_callback)

# Actual repair duration useful as a predictor known after job; exclude from model to avoid post-outcome leakage.
actual_duration_min = np.clip(25 + 22*job_complexity + 2.2*door_age + 12*(parts_available!='Full') - 2.0*technician_experience + rng.normal(0,18,N),15,360).round().astype(int)

# Service cost: included for reporting, not used in callback model because it is usually finalized after service.
service_cost = np.clip(95 + 55*job_complexity + 18*previous_repairs + 45*(service_type=='Installation') + rng.normal(0,35,N),75,1200).round(2)

df = pd.DataFrame({
    'job_id':[f'J{i:06d}' for i in range(1,N+1)],
    'branch_id':rng.choice(branches,N),
    'service_type':service_type,
    'problem_type':problem_type,
    'door_gate_type':door_type,
    'door_gate_age_years':door_age,
    'previous_repairs':previous_repairs,
    'prior_callbacks':prior_callbacks,
    'technician_experience_years':technician_experience,
    'technician_skill_level':technician_skill,
    'required_skill_level':required_skill,
    'distance_km':distance_km,
    'traffic_level':traffic_level,
    'urgency':urgency,
    'parts_availability':parts_available,
    'remote_diagnosis_quality':remote_diagnosis_quality,
    'customer_described_symptoms':customer_described_symptoms,
    'first_appointment_of_day':first_appointment_of_day,
    'weekend':weekend,
    'time_slot':time_slot,
    'job_complexity':job_complexity,
    'actual_duration_min':actual_duration_min,
    'service_cost_usd':service_cost,
    'callback_repeat_visit':callback
})
df.to_csv(DATA_DIR/'garage_callback_synthetic_seed42.csv', index=False)

# Data dictionary
rows = [
    ('job_id','Unique synthetic service-job ID','Identifier'),('branch_id','Synthetic branch ID','Input'),
    ('service_type','High-level service category','Input'),('problem_type','Reported/diagnosed issue type','Input'),
    ('door_gate_type','Garage door or gate configuration','Input'),('door_gate_age_years','Equipment age in years','Input'),
    ('previous_repairs','Number of previous repairs','Input'),('prior_callbacks','Previous repeat visits for this equipment/customer','Input'),
    ('technician_experience_years','Assigned technician experience','Input'),('technician_skill_level','Technician skill level 1-5','Input'),
    ('required_skill_level','Estimated skill requirement 1-5','Input'),('distance_km','Technician travel distance','Input'),
    ('traffic_level','Expected travel traffic','Input'),('urgency','Service urgency','Input'),
    ('parts_availability','Required-parts readiness: Unavailable/Partial/Full','Input'),('remote_diagnosis_quality','Pre-dispatch diagnosis quality 1-5','Input'),
    ('customer_described_symptoms','Whether customer supplied useful symptom details','Input'),('first_appointment_of_day','Whether job is tech first appointment','Input'),
    ('weekend','Weekend service flag','Input'),('time_slot','Scheduled service period','Input'),
    ('job_complexity','Estimated complexity 1-5','Input'),('actual_duration_min','Observed post-job duration; excluded to prevent leakage','Post-outcome'),
    ('service_cost_usd','Observed post-job service charge; excluded to prevent leakage','Post-outcome'),
    ('callback_repeat_visit','Target: 1 repeat visit, 0 first-time fix','Target')
]
pd.DataFrame(rows, columns=['column','meaning','role']).to_csv(DATA_DIR/'data_dictionary.csv',index=False)

# -------------------------
# 2. Train/validation/test modeling
# -------------------------
TARGET='callback_repeat_visit'
DROP=['job_id','actual_duration_min','service_cost_usd',TARGET]
X=df.drop(columns=DROP)
y=df[TARGET]

# 70% train, 15% validation, 15% final test. Stratification preserves callback prevalence.
X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30,stratify=y,random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=0.50,stratify=y_temp,random_state=SEED)

cat_cols=X.select_dtypes(include='object').columns.tolist()
num_cols=[c for c in X.columns if c not in cat_cols]
preprocessor=ColumnTransformer([
    ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num_cols),
    ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat_cols)
])
X_train_t=preprocessor.fit_transform(X_train)
X_val_t=preprocessor.transform(X_val)
X_test_t=preprocessor.transform(X_test)
joblib.dump(preprocessor, MODELS_DIR/'preprocessor.joblib')

positive_weight=float((y_train==0).sum()/(y_train==1).sum())
models={
    'Logistic Regression':LogisticRegression(max_iter=2000,class_weight='balanced',random_state=SEED),
    'Decision Tree':DecisionTreeClassifier(max_depth=8,min_samples_leaf=35,class_weight='balanced',random_state=SEED),
    'Random Forest':RandomForestClassifier(n_estimators=350,max_depth=14,min_samples_leaf=8,class_weight='balanced',n_jobs=-1,random_state=SEED),
    'XGBoost':XGBClassifier(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.9,colsample_bytree=0.85,eval_metric='logloss',random_state=SEED,n_jobs=-1,scale_pos_weight=positive_weight),
    'CatBoost':CatBoostClassifier(iterations=500,depth=6,learning_rate=0.04,loss_function='Logloss',verbose=False,random_seed=SEED,auto_class_weights='Balanced'),
    'Neural Network':MLPClassifier(hidden_layer_sizes=(64,32),alpha=0.001,learning_rate_init=0.001,max_iter=300,early_stopping=True,random_state=SEED)
}

results=[]
predictions=pd.DataFrame({'job_id':df.loc[X_test.index,'job_id'].values,'actual_callback':y_test.values})
fitted={}; thresholds_by_model={}; validation_rows=[]

for name, model in models.items():
    model.fit(X_train_t,y_train)
    fitted[name]=model
    joblib.dump(model,MODELS_DIR/(name.lower().replace(' ','_')+'.joblib'))

    # Choose threshold on validation only, maximizing F1 for callback class.
    val_prob=model.predict_proba(X_val_t)[:,1]
    threshold_grid=np.arange(0.10,0.91,0.01)
    val_scores=[]
    for t in threshold_grid:
        vp=(val_prob>=t).astype(int)
        val_scores.append((t,precision_score(y_val,vp,zero_division=0),recall_score(y_val,vp,zero_division=0),f1_score(y_val,vp,zero_division=0)))
    val_table=pd.DataFrame(val_scores,columns=['threshold','precision','recall','f1'])
    chosen=float(val_table.loc[val_table.f1.idxmax(),'threshold'])
    thresholds_by_model[name]=chosen
    validation_rows.append({'Model':name,'Chosen_Threshold':chosen,'Validation_F1':float(val_table.f1.max())})

    # Final untouched test evaluation using the validation-chosen threshold.
    prob=model.predict_proba(X_test_t)[:,1]
    pred=(prob>=chosen).astype(int)
    cm=confusion_matrix(y_test,pred)
    results.append({
        'Model':name,'Threshold':chosen,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred,zero_division=0),
        'Recall':recall_score(y_test,pred,zero_division=0),'F1':f1_score(y_test,pred,zero_division=0),
        'ROC_AUC':roc_auc_score(y_test,prob),'PR_AUC':average_precision_score(y_test,prob),
        'TN':cm[0,0],'FP':cm[0,1],'FN':cm[1,0],'TP':cm[1,1]
    })
    predictions[name+'_probability']=prob
    predictions[name+'_prediction']=pred

metrics=pd.DataFrame(results).sort_values(['F1','ROC_AUC'],ascending=False)
metrics.to_csv(OUT_DIR/'model_metrics.csv',index=False)
pd.DataFrame(validation_rows).to_csv(OUT_DIR/'validation_thresholds.csv',index=False)
predictions.to_csv(OUT_DIR/'test_set_predictions.csv',index=False)

best_f1_name=metrics.sort_values('F1',ascending=False).iloc[0]['Model']
best_auc_name=metrics.sort_values('ROC_AUC',ascending=False).iloc[0]['Model']
deployment_name=best_f1_name
deployment_model=fitted[deployment_name]
deployment_threshold=thresholds_by_model[deployment_name]

# Threshold curve for deployment model is based on validation set, not the final test set.
deploy_val_prob=deployment_model.predict_proba(X_val_t)[:,1]
th_rows=[]
for t in np.arange(0.10,0.91,0.01):
    p=(deploy_val_prob>=t).astype(int)
    th_rows.append((t,precision_score(y_val,p,zero_division=0),recall_score(y_val,p,zero_division=0),f1_score(y_val,p,zero_division=0)))
th=pd.DataFrame(th_rows,columns=['threshold','precision','recall','f1'])
th.to_csv(OUT_DIR/'threshold_analysis_deployment_model.csv',index=False)

# Four illustrative unseen business scenarios.
scenarios=pd.DataFrame([
    {'branch_id':'B03','service_type':'Garage Door Repair','problem_type':'Broken Spring','door_gate_type':'Double Steel','door_gate_age_years':15.0,'previous_repairs':3,'prior_callbacks':1,'technician_experience_years':1.2,'technician_skill_level':2,'required_skill_level':4,'distance_km':16.0,'traffic_level':'High','urgency':'Emergency','parts_availability':'Partial','remote_diagnosis_quality':2,'customer_described_symptoms':1,'first_appointment_of_day':0,'weekend':1,'time_slot':'Evening','job_complexity':4},
    {'branch_id':'B07','service_type':'Garage Door Opener','problem_type':'Remote/Keypad Issue','door_gate_type':'Single Steel','door_gate_age_years':4.0,'previous_repairs':0,'prior_callbacks':0,'technician_experience_years':7.5,'technician_skill_level':4,'required_skill_level':2,'distance_km':5.0,'traffic_level':'Low','urgency':'Normal','parts_availability':'Full','remote_diagnosis_quality':5,'customer_described_symptoms':1,'first_appointment_of_day':1,'weekend':0,'time_slot':'Morning','job_complexity':1},
    {'branch_id':'B10','service_type':'Automatic Gate Repair','problem_type':'Gate Motor Failure','door_gate_type':'Sliding Gate','door_gate_age_years':11.0,'previous_repairs':2,'prior_callbacks':0,'technician_experience_years':4.0,'technician_skill_level':4,'required_skill_level':5,'distance_km':8.5,'traffic_level':'Medium','urgency':'High','parts_availability':'Full','remote_diagnosis_quality':4,'customer_described_symptoms':1,'first_appointment_of_day':0,'weekend':0,'time_slot':'Afternoon','job_complexity':5},
    {'branch_id':'B05','service_type':'Garage Door Repair','problem_type':'Sensor Failure','door_gate_type':'Double Steel','door_gate_age_years':9.0,'previous_repairs':1,'prior_callbacks':0,'technician_experience_years':0.8,'technician_skill_level':2,'required_skill_level':3,'distance_km':22.0,'traffic_level':'High','urgency':'High','parts_availability':'Unavailable','remote_diagnosis_quality':2,'customer_described_symptoms':0,'first_appointment_of_day':0,'weekend':0,'time_slot':'Afternoon','job_complexity':3},
])
scenario_t=preprocessor.transform(scenarios[X.columns])
for name,model in fitted.items():
    scenarios[name+'_probability']=model.predict_proba(scenario_t)[:,1]
scenarios['deployment_model']=deployment_name
scenarios['decision_threshold']=deployment_threshold
scenarios['callback_prediction']=(scenarios[deployment_name+'_probability']>=deployment_threshold).astype(int)
scenarios.to_csv(OUT_DIR/'sample_new_job_predictions.csv',index=False)

# -------------------------
# 3. Plots
# -------------------------
plt.figure(figsize=(10,5.5))
plotm=metrics.set_index('Model')[['Accuracy','Precision','Recall','F1','ROC_AUC']]
plotm.plot(kind='bar',ax=plt.gca())
plt.ylim(0,1); plt.ylabel('Score'); plt.title('Callback Prediction Model Comparison')
plt.xticks(rotation=25,ha='right'); plt.legend(loc='lower right',ncol=2)
plt.tight_layout(); plt.savefig(PLOTS_DIR/'model_comparison.png',dpi=200); plt.close()

plt.figure(figsize=(7,6))
for name in fitted:
    RocCurveDisplay.from_predictions(y_test,predictions[name+'_probability'],name=name,ax=plt.gca())
plt.plot([0,1],[0,1],'--',linewidth=1); plt.title('ROC Curves: Final Test Set')
plt.tight_layout(); plt.savefig(PLOTS_DIR/'roc_curves.png',dpi=200); plt.close()

plt.figure(figsize=(7,6))
for name in fitted:
    PrecisionRecallDisplay.from_predictions(y_test,predictions[name+'_probability'],name=name,ax=plt.gca())
plt.title('Precision-Recall Curves: Final Test Set')
plt.tight_layout(); plt.savefig(PLOTS_DIR/'precision_recall_curves.png',dpi=200); plt.close()

plt.figure(figsize=(8,5))
plt.plot(th.threshold,th.precision,label='Precision'); plt.plot(th.threshold,th.recall,label='Recall'); plt.plot(th.threshold,th.f1,label='F1')
plt.axvline(deployment_threshold,linestyle='--',label=f'Chosen threshold={deployment_threshold:.2f}')
plt.xlabel('Decision threshold'); plt.ylabel('Score'); plt.ylim(0,1); plt.title(f'Validation Threshold Trade-off: {deployment_name}')
plt.legend(); plt.tight_layout(); plt.savefig(PLOTS_DIR/'threshold_tradeoff.png',dpi=200); plt.close()

# Feature importance for deployment model.
feature_names=preprocessor.get_feature_names_out()
if hasattr(deployment_model,'feature_importances_'):
    importance=np.asarray(deployment_model.feature_importances_)
elif hasattr(deployment_model,'coef_'):
    importance=np.abs(np.asarray(deployment_model.coef_).ravel())
else:
    perm=permutation_importance(deployment_model,X_test_t,y_test,scoring='roc_auc',n_repeats=3,random_state=SEED,n_jobs=-1)
    importance=perm.importances_mean
imp=pd.DataFrame({'Feature':feature_names,'Importance':importance}).sort_values('Importance',ascending=False)
imp.to_csv(OUT_DIR/'feature_importance.csv',index=False)
plt.figure(figsize=(9,6))
top=imp.head(15).sort_values('Importance')
plt.barh(top.Feature.str.replace('num__','',regex=False).str.replace('cat__','',regex=False),top.Importance)
plt.xlabel('Importance'); plt.title(f'Top Predictive Features: {deployment_name}')
plt.tight_layout(); plt.savefig(PLOTS_DIR/'feature_importance.png',dpi=200); plt.close()

rate_parts=df.groupby('parts_availability')[TARGET].mean().reindex(['Full','Partial','Unavailable'])
plt.figure(figsize=(6,4)); rate_parts.mul(100).plot(kind='bar'); plt.ylabel('Callback rate (%)'); plt.title('Callback Rate by Parts Availability'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(PLOTS_DIR/'callback_by_parts.png',dpi=200); plt.close()

rate_skill=df.assign(skill_gap=df.required_skill_level-df.technician_skill_level).groupby('skill_gap')[TARGET].agg(['mean','count']).reset_index()
rate_skill=rate_skill[rate_skill['count']>=50]
plt.figure(figsize=(7,4.5)); plt.plot(rate_skill.skill_gap,rate_skill['mean']*100,marker='o'); plt.xlabel('Required skill - technician skill'); plt.ylabel('Callback rate (%)'); plt.title('Callback Risk Rises with Technician Skill Gap'); plt.tight_layout(); plt.savefig(PLOTS_DIR/'callback_by_skill_gap.png',dpi=200); plt.close()

summary={
    'seed':SEED,'rows':N,'callback_rate':float(y.mean()),'train_rows':len(X_train),'validation_rows':len(X_val),'test_rows':len(X_test),
    'best_f1_model':best_f1_name,'best_roc_auc_model':best_auc_name,'deployment_model':deployment_name,
    'deployment_threshold_from_validation':deployment_threshold,
    'deployment_test_metrics':metrics.loc[metrics.Model==deployment_name].iloc[0].to_dict()
}
with open(OUT_DIR/'run_summary.json','w') as f: json.dump(summary,f,indent=2,default=float)

print('Synthetic dataset created:', DATA_DIR/'garage_callback_synthetic_seed42.csv')
print('Callback rate:', round(y.mean(),4))
print(f'Split: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}')
print('\nFinal test results using validation-selected thresholds:')
print(metrics[['Model','Threshold','Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC']].to_string(index=False))
print('\nBest F1 model:',best_f1_name)
print('Best ROC-AUC model:',best_auc_name)
print('Deployment model:',deployment_name,'threshold=',round(deployment_threshold,2))
print('\nScenario predictions from deployment model:')
print(scenarios[[deployment_name+'_probability','callback_prediction']].to_string(index=False))
