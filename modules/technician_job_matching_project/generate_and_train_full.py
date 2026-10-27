from pathlib import Path
import math, warnings, shutil, zipfile, json
import numpy as np
import pandas as pd

BASE=Path('/mnt/data/technician_job_matching_project')
if BASE.exists(): shutil.rmtree(BASE)
for s in ['data','models','outputs','plots']:(BASE/s).mkdir(parents=True,exist_ok=True)
SEED=42
rng=np.random.default_rng(SEED)
N_TECH=40; N_JOBS=2500; CAND=8
branches=['North','South','East','West']
service_types=['Garage Door Repair','Automatic Gate Repair','Garage Door Opener','New Installation']
part_types=['spring','cable','roller','sensor','motor','control_board','remote','none']

# technicians
techs=pd.DataFrame({
 'technician_id':[f'T{i:03d}' for i in range(1,N_TECH+1)],
 'branch':rng.choice(branches,N_TECH),
 'experience_years':np.round(rng.gamma(2.2,2.2,N_TECH).clip(.5,15),1),
 'garage_mechanical_skill':rng.integers(1,6,N_TECH),
 'gate_electrical_skill':rng.integers(1,6,N_TECH),
 'opener_motor_skill':rng.integers(1,6,N_TECH),
 'installation_skill':rng.integers(1,6,N_TECH),
 'customer_rating':np.round(rng.normal(4.45,.35,N_TECH).clip(3.2,5),2),
 'historical_first_fix_rate':np.round(rng.beta(18,4,N_TECH).clip(.55,.98),3),
 'callback_rate':np.round(rng.beta(3,20,N_TECH).clip(.02,.35),3),
 'current_workload_ratio':np.round(rng.beta(4,3,N_TECH).clip(.05,.98),3),
 'available_in_minutes':rng.integers(0,181,N_TECH),
 'tech_lat':rng.uniform(0,50,N_TECH), 'tech_lon':rng.uniform(0,50,N_TECH)
})
for p in ['spring','cable','roller','sensor','motor','control_board','remote']:
    techs[f'has_{p}']=rng.binomial(1,.72,N_TECH)

def choose_problem(service):
    if service=='Garage Door Repair': return rng.choice(['Broken Spring','Cable Failure','Off Track Door','Sensor Failure','Routine Maintenance'],p=[.29,.20,.18,.18,.15])
    if service=='Automatic Gate Repair': return rng.choice(['Gate Motor Failure','Sensor Failure','Control Board Failure','Remote/Keypad Issue','Routine Maintenance'],p=[.30,.18,.22,.15,.15])
    if service=='Garage Door Opener': return rng.choice(['Opener Motor Failure','Sensor Failure','Control Board Failure','Remote/Keypad Issue'],p=[.38,.22,.22,.18])
    return 'New Installation'

problem_to_part={'Broken Spring':'spring','Cable Failure':'cable','Off Track Door':'roller','Sensor Failure':'sensor','Opener Motor Failure':'motor','Gate Motor Failure':'motor','Control Board Failure':'control_board','Remote/Keypad Issue':'remote','Routine Maintenance':'none','New Installation':'none'}
problem_skill={'Broken Spring':4,'Cable Failure':3,'Off Track Door':4,'Sensor Failure':3,'Opener Motor Failure':4,'Gate Motor Failure':5,'Control Board Failure':5,'Remote/Keypad Issue':2,'Routine Maintenance':2,'New Installation':4}
problem_duration={'Broken Spring':85,'Cable Failure':70,'Off Track Door':95,'Sensor Failure':45,'Opener Motor Failure':100,'Gate Motor Failure':125,'Control Board Failure':105,'Remote/Keypad Issue':35,'Routine Maintenance':50,'New Installation':210}
services=rng.choice(service_types,N_JOBS,p=[.42,.25,.23,.10])
problems=[choose_problem(s) for s in services]
jobs=pd.DataFrame({
 'job_id':[f'J{i:05d}' for i in range(1,N_JOBS+1)], 'branch':rng.choice(branches,N_JOBS),
 'service_type':services, 'problem_type':problems,
 'urgency':rng.choice(['Low','Normal','High','Emergency'],N_JOBS,p=[.12,.48,.25,.15]),
 'equipment_age_years':np.round(rng.gamma(2.4,3.0,N_JOBS).clip(0,25),1),
 'previous_repairs':rng.poisson(1.2,N_JOBS).clip(0,7),
 'estimated_duration_minutes':[int(max(20,rng.normal(problem_duration[p],problem_duration[p]*.16))) for p in problems],
 'required_skill':[int(np.clip(problem_skill[p]+rng.choice([-1,0,0,0,1]),1,5)) for p in problems],
 'required_part':[problem_to_part[p] for p in problems],
 'job_lat':rng.uniform(0,50,N_JOBS), 'job_lon':rng.uniform(0,50,N_JOBS)
})
urgency_num={'Low':1,'Normal':2,'High':3,'Emergency':4}
rows=[]
for _,job in jobs.iterrows():
    for ti in rng.choice(N_TECH,CAND,replace=False):
        tech=techs.iloc[ti]
        if job.service_type=='Garage Door Repair': rel=tech.garage_mechanical_skill
        elif job.service_type=='Automatic Gate Repair': rel=tech.gate_electrical_skill
        elif job.service_type=='Garage Door Opener': rel=tech.opener_motor_skill
        else: rel=tech.installation_skill
        dist=math.hypot(job.job_lat-tech.tech_lat,job.job_lon-tech.tech_lon)*1.35
        travel=dist/42*60*rng.uniform(1.0,1.6)
        inv=1 if job.required_part=='none' else int(tech[f'has_{job.required_part}'])
        same=int(job.branch==tech.branch)
        gap=float(rel-job.required_skill); ratio=float(np.clip(rel/job.required_skill,0,2))
        urg=urgency_num[job.urgency]
        z=(2.10*gap+2.35*inv+1.10*same+.16*tech.experience_years+1.55*(tech.customer_rating-4)+4.30*(tech.historical_first_fix_rate-.75)-4.10*tech.callback_rate-1.65*tech.current_workload_ratio-.020*dist-.012*tech.available_in_minutes-.004*job.estimated_duration_minutes*tech.current_workload_ratio+.55*(inv*max(gap,-1))+.38*(urg*same)-.50*((urg>=3) and tech.available_in_minutes>90)-.45*((job.equipment_age_years>15) and rel<4))
        base=100/(1+np.exp(-.55*z)); true=float(np.clip(base+rng.normal(0,5.5),0,100))
        success_prob=np.clip(.20+.0065*true+.11*inv+.035*max(gap,0)-.05*(job.equipment_age_years>15),.05,.98)
        ftf=int(rng.random()<success_prob)
        sla={1:240,2:180,3:120,4:75}[urg]; response=float(tech.available_in_minutes+travel); sla_met=int(response<=sla)
        sat=float(np.clip(3.1+.012*true+.55*ftf+.2*sla_met-.003*response+rng.normal(0,.28),1,5))
        rows.append({
          'job_id':job.job_id,'technician_id':tech.technician_id,'job_branch':job.branch,'technician_branch':tech.branch,
          'service_type':job.service_type,'problem_type':job.problem_type,'urgency':job.urgency,'equipment_age_years':job.equipment_age_years,
          'previous_repairs':job.previous_repairs,'estimated_duration_minutes':job.estimated_duration_minutes,'required_skill':job.required_skill,
          'required_part':job.required_part,'experience_years':tech.experience_years,'relevant_skill':int(rel),'customer_rating':tech.customer_rating,
          'historical_first_fix_rate':tech.historical_first_fix_rate,'callback_rate':tech.callback_rate,'current_workload_ratio':tech.current_workload_ratio,
          'available_in_minutes':int(tech.available_in_minutes),'distance_km':round(dist,2),'travel_time_minutes':round(travel,1),'inventory_match':inv,
          'same_branch':same,'skill_gap':gap,'skill_match_ratio':round(ratio,3),'true_suitability_score':round(true,3),
          'sim_first_time_fix':ftf,'sim_sla_met':sla_met,'sim_response_time_minutes':round(response,1),'sim_customer_satisfaction':round(sat,2)})
pairs=pd.DataFrame(rows)
techs.to_csv(BASE/'data'/'technicians_seed42.csv',index=False); jobs.to_csv(BASE/'data'/'jobs_seed42.csv',index=False); pairs.to_csv(BASE/'data'/'technician_job_candidates_seed42.csv',index=False)

# ML
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import joblib

ids=jobs.job_id.to_numpy(); train_ids,test_ids=train_test_split(ids,test_size=.20,random_state=SEED); train_ids,val_ids=train_test_split(train_ids,test_size=.1875,random_state=SEED)
features=['job_branch','technician_branch','service_type','problem_type','urgency','required_part','equipment_age_years','previous_repairs','estimated_duration_minutes','required_skill','experience_years','relevant_skill','customer_rating','historical_first_fix_rate','callback_rate','current_workload_ratio','available_in_minutes','distance_km','travel_time_minutes','inventory_match','same_branch','skill_gap','skill_match_ratio']
cat=['job_branch','technician_branch','service_type','problem_type','urgency','required_part']; num=[c for c in features if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={'Linear Regression':LinearRegression(),'Decision Tree':DecisionTreeRegressor(max_depth=9,min_samples_leaf=10,random_state=SEED),'Random Forest':RandomForestRegressor(n_estimators=220,max_depth=15,min_samples_leaf=3,random_state=SEED,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(n_estimators=180,learning_rate=.05,max_depth=3,random_state=SEED),'Neural Network':MLPRegressor(hidden_layer_sizes=(64,32),max_iter=350,early_stopping=True,random_state=SEED)}
try:
 from xgboost import XGBRegressor
 models['XGBoost']=XGBRegressor(n_estimators=300,max_depth=5,learning_rate=.04,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=SEED,n_jobs=-1)
except Exception as e: print('XGBoost unavailable',e)
train=pairs[pairs.job_id.isin(train_ids)]; test=pairs[pairs.job_id.isin(test_ids)]
metrics=[]; fitted={}; preds={}
for name,m in models.items():
    pipe=Pipeline([('prep',pre),('model',m)])
    with warnings.catch_warnings(): warnings.simplefilter('ignore'); pipe.fit(train[features],train.true_suitability_score)
    pr=np.clip(pipe.predict(test[features]),0,100); fitted[name]=pipe; preds[name]=pr
    metrics.append({'Model':name,'MAE':mean_absolute_error(test.true_suitability_score,pr),'RMSE':mean_squared_error(test.true_suitability_score,pr)**.5,'R2':r2_score(test.true_suitability_score,pr)})
    joblib.dump(pipe,BASE/'models'/(name.lower().replace(' ','_')+'.joblib'))
metrics_df=pd.DataFrame(metrics).sort_values(['MAE','RMSE']).reset_index(drop=True); metrics_df.to_csv(BASE/'outputs'/'model_metrics.csv',index=False)
best_name=metrics_df.iloc[0].Model; best=fitted[best_name]
scored=test.copy(); scored['predicted_suitability_score']=np.round(preds[best_name],3); scored.to_csv(BASE/'outputs'/'test_candidate_predictions.csv',index=False)

# strategy comparisons
ml=scored.loc[scored.groupby('job_id').predicted_suitability_score.idxmax()].copy(); oracle=scored.loc[scored.groupby('job_id').true_suitability_score.idxmax()].copy(); nearest=scored.loc[scored.groupby('job_id').sim_response_time_minutes.idxmin()].copy()
simple=scored.copy(); simple['rule_score']=30*np.clip(simple.skill_match_ratio,0,1.25)/1.25+25*simple.inventory_match+12*simple.historical_first_fix_rate+10*(simple.customer_rating/5)+8*simple.same_branch+8*(1-simple.current_workload_ratio)+7*(1-np.clip(simple.distance_km/80,0,1)); rule=simple.loc[simple.groupby('job_id').rule_score.idxmax()].copy(); rand=scored.groupby('job_id',group_keys=False).sample(n=1,random_state=SEED)
oracle_map=oracle.set_index('job_id').true_suitability_score
def kpi(name,df):
 reg=oracle_map.loc[df.job_id].to_numpy()-df.true_suitability_score.to_numpy()
 return {'Strategy':name,'Avg True Suitability':df.true_suitability_score.mean(),'First-Time Fix Rate':df.sim_first_time_fix.mean(),'SLA Met Rate':df.sim_sla_met.mean(),'Avg Response Time (min)':df.sim_response_time_minutes.mean(),'Avg Customer Satisfaction':df.sim_customer_satisfaction.mean(),'Avg Regret vs Oracle':reg.mean()}
kpis=pd.DataFrame([kpi('Random Assignment',rand),kpi('Nearest/Available',nearest),kpi('Weighted Rule',rule),kpi(f'ML Ranking ({best_name})',ml),kpi('Oracle Upper Bound',oracle)])
kpis.to_csv(BASE/'outputs'/'dispatch_strategy_kpis.csv',index=False); ml.to_csv(BASE/'outputs'/'ml_selected_technicians.csv',index=False)
oracle_best=oracle.set_index('job_id').technician_id; ml_best=ml.set_index('job_id').technician_id; top1=float((oracle_best.loc[ml_best.index]==ml_best).mean()); top3=float(np.mean([oracle_best.loc[j] in set(g.nlargest(3,'predicted_suitability_score').technician_id) for j,g in scored.groupby('job_id')]))

# sample new job score all techs
new={'job_branch':'East','service_type':'Automatic Gate Repair','problem_type':'Gate Motor Failure','urgency':'Emergency','equipment_age_years':12.0,'previous_repairs':3,'estimated_duration_minutes':125,'required_skill':5,'required_part':'motor','job_lat':25.0,'job_lon':28.0}
r=[]
for _,t in techs.iterrows():
 rel=t.gate_electrical_skill; dist=math.hypot(new['job_lat']-t.tech_lat,new['job_lon']-t.tech_lon)*1.35; travel=dist/42*60*1.25; inv=int(t.has_motor); same=int(t.branch==new['job_branch'])
 r.append({'job_branch':new['job_branch'],'technician_branch':t.branch,'service_type':new['service_type'],'problem_type':new['problem_type'],'urgency':new['urgency'],'required_part':new['required_part'],'equipment_age_years':new['equipment_age_years'],'previous_repairs':new['previous_repairs'],'estimated_duration_minutes':new['estimated_duration_minutes'],'required_skill':new['required_skill'],'experience_years':t.experience_years,'relevant_skill':rel,'customer_rating':t.customer_rating,'historical_first_fix_rate':t.historical_first_fix_rate,'callback_rate':t.callback_rate,'current_workload_ratio':t.current_workload_ratio,'available_in_minutes':t.available_in_minutes,'distance_km':round(dist,2),'travel_time_minutes':round(travel,1),'inventory_match':inv,'same_branch':same,'skill_gap':float(rel-new['required_skill']),'skill_match_ratio':round(rel/new['required_skill'],3),'technician_id':t.technician_id})
newdf=pd.DataFrame(r); newdf['predicted_suitability_score']=np.clip(best.predict(newdf[features]),0,100); newdf=newdf.sort_values('predicted_suitability_score',ascending=False).reset_index(drop=True); newdf['rank']=np.arange(1,len(newdf)+1); newdf.to_csv(BASE/'outputs'/'sample_new_job_rankings.csv',index=False)

# feature importance
try:
 names=best.named_steps['prep'].get_feature_names_out(); model=best.named_steps['model']; vals=model.feature_importances_ if hasattr(model,'feature_importances_') else np.abs(np.asarray(model.coef_).ravel()) if hasattr(model,'coef_') else None
 if vals is not None and len(vals)==len(names): pd.DataFrame({'feature':names,'importance':vals}).sort_values('importance',ascending=False).to_csv(BASE/'outputs'/'feature_importance.csv',index=False)
except Exception as e: print('importance skipped',e)

# plots
import matplotlib.pyplot as plt
p=metrics_df.sort_values('MAE'); plt.figure(figsize=(9,5)); plt.barh(p.Model,p.MAE); plt.xlabel('MAE (lower is better)'); plt.title('Technician Suitability Model Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'model_comparison_mae.png',dpi=180); plt.close()
p=metrics_df.sort_values('R2'); plt.figure(figsize=(9,5)); plt.barh(p.Model,p.R2); plt.xlabel('R² (higher is better)'); plt.title('Model R² Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'model_comparison_r2.png',dpi=180); plt.close()
p=kpis.sort_values('Avg True Suitability'); plt.figure(figsize=(10,5)); plt.barh(p.Strategy,p['Avg True Suitability']); plt.xlabel('Average true suitability'); plt.title('Dispatch Strategy Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'dispatch_strategy_suitability.png',dpi=180); plt.close()
p=kpis.sort_values('First-Time Fix Rate'); plt.figure(figsize=(10,5)); plt.barh(p.Strategy,p['First-Time Fix Rate']*100); plt.xlabel('First-time fix rate (%)'); plt.title('First-Time Fix by Dispatch Strategy'); plt.tight_layout(); plt.savefig(BASE/'plots'/'dispatch_strategy_first_fix.png',dpi=180); plt.close()
s=scored.sample(min(1800,len(scored)),random_state=SEED); plt.figure(figsize=(6.5,6)); plt.scatter(s.true_suitability_score,s.predicted_suitability_score,alpha=.25,s=12); plt.plot([0,100],[0,100],'--'); plt.xlabel('True suitability'); plt.ylabel('Predicted suitability'); plt.title(f'Actual vs Predicted ({best_name})'); plt.tight_layout(); plt.savefig(BASE/'plots'/'actual_vs_predicted.png',dpi=180); plt.close()
top=newdf.head(10).sort_values('predicted_suitability_score'); plt.figure(figsize=(9,5)); plt.barh(top.technician_id,top.predicted_suitability_score); plt.xlabel('Predicted suitability'); plt.title('Top 10 Technicians for Sample Emergency Gate Job'); plt.tight_layout(); plt.savefig(BASE/'plots'/'sample_job_top10.png',dpi=180); plt.close()

# dictionary
pd.DataFrame([
('job_id','Unique service job ID'),('technician_id','Candidate technician ID'),('service_type','Garage/gate service category'),('problem_type','Specific service problem'),('urgency','Low, Normal, High, Emergency'),('required_skill','Required skill level 1-5'),('required_part','Likely needed part'),('experience_years','Technician experience'),('relevant_skill','Technician skill relevant to job 1-5'),('historical_first_fix_rate','Technician historical one-visit success rate'),('callback_rate','Technician historical repeat-visit rate'),('current_workload_ratio','Current workload 0-1'),('available_in_minutes','Minutes until available'),('distance_km','Technician-to-job distance'),('travel_time_minutes','Estimated travel time'),('inventory_match','1 if required part is carried'),('same_branch','1 if same branch'),('skill_gap','Relevant skill minus required skill'),('skill_match_ratio','Relevant skill / required skill'),('true_suitability_score','Synthetic target 0-100'),('sim_first_time_fix','Simulated first-time-fix outcome'),('sim_sla_met','Simulated SLA outcome'),('sim_response_time_minutes','Availability + travel'),('sim_customer_satisfaction','Simulated satisfaction 1-5')],columns=['column','description']).to_csv(BASE/'data'/'data_dictionary.csv',index=False)

# executable prediction helper
pred_script='''from pathlib import Path\nimport pandas as pd, joblib\nBASE=Path(__file__).resolve().parent\nranked=pd.read_csv(BASE/"outputs"/"sample_new_job_rankings.csv")\nprint(ranked[["rank","technician_id","predicted_suitability_score","relevant_skill","inventory_match","distance_km","available_in_minutes"]].head(10).to_string(index=False))\n'''
(BASE/'predict_new_job.py').write_text(pred_script)
(BASE/'requirements.txt').write_text('numpy\npandas\nscikit-learn\nmatplotlib\njoblib\nxgboost\njupyter\n')

bestrow=metrics_df.iloc[0]; mlk=kpis[kpis.Strategy.str.startswith('ML Ranking')].iloc[0]; nk=kpis[kpis.Strategy=='Nearest/Available'].iloc[0]; rk=kpis[kpis.Strategy=='Weighted Rule'].iloc[0]
readme=f'''# Technician-Job Matching / Suitability Score\n\n## Goal\nRank candidate technicians for each garage-door or automatic-gate service job and recommend the best technician.\n\n## Reproducible synthetic dataset\nSeed: **{SEED}**\n\n- Jobs: {N_JOBS:,}\n- Technicians: {N_TECH}\n- Candidate technicians per job: {CAND}\n- Candidate rows: {len(pairs):,}\n\nThe target `true_suitability_score` (0-100) reflects realistic factors: skill fit, parts availability, branch match, experience, first-time-fix history, callback history, workload, distance, availability, urgency interactions, and random noise.\n\n## Models\nLinear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, Neural Network.\n\n## Best model\n**{best_name}**\n- MAE: {bestrow.MAE:.3f}\n- RMSE: {bestrow.RMSE:.3f}\n- R²: {bestrow.R2:.3f}\n\n## Ranking quality\n- Exact Top-1 agreement with hidden oracle: {top1*100:.1f}%\n- Hidden oracle technician appears in ML Top-3: {top3*100:.1f}%\n\n## Operational comparison\nNearest/Available: suitability {nk['Avg True Suitability']:.2f}, first-time-fix {nk['First-Time Fix Rate']*100:.1f}%, SLA {nk['SLA Met Rate']*100:.1f}%, response {nk['Avg Response Time (min)']:.1f} min.\n\nWeighted Rule: suitability {rk['Avg True Suitability']:.2f}, first-time-fix {rk['First-Time Fix Rate']*100:.1f}%, SLA {rk['SLA Met Rate']*100:.1f}%, response {rk['Avg Response Time (min)']:.1f} min.\n\nML Ranking: suitability {mlk['Avg True Suitability']:.2f}, first-time-fix {mlk['First-Time Fix Rate']*100:.1f}%, SLA {mlk['SLA Met Rate']*100:.1f}%, response {mlk['Avg Response Time (min)']:.1f} min.\n\n## Files\n- `data/technicians_seed42.csv`\n- `data/jobs_seed42.csv`\n- `data/technician_job_candidates_seed42.csv`\n- `outputs/model_metrics.csv`\n- `outputs/dispatch_strategy_kpis.csv`\n- `outputs/sample_new_job_rankings.csv`\n- `models/*.joblib`\n- `plots/*.png`\n\n## Academic note\nThis is synthetic field-service data, not real Precision Garage Door Service operational data. Present it as a reproducible simulation/prototype unless real company data becomes available.\n'''
(BASE/'README.md').write_text(readme)
summary={'seed':SEED,'jobs':N_JOBS,'technicians':N_TECH,'candidate_rows':len(pairs),'best_model':best_name,'best_model_metrics':{'mae':float(bestrow.MAE),'rmse':float(bestrow.RMSE),'r2':float(bestrow.R2)},'top1_oracle_match':top1,'top3_oracle_capture':top3}
(BASE/'outputs'/'run_summary.json').write_text(json.dumps(summary,indent=2))
# Notebook summary
try:
 import nbformat as nbf
 nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell('# Technician-Job Matching ML Project\nReproducible synthetic project, seed = 42.'),nbf.v4.new_code_cell('import pandas as pd\npairs=pd.read_csv("data/technician_job_candidates_seed42.csv")\npairs.head()'),nbf.v4.new_code_cell('pd.read_csv("outputs/model_metrics.csv")'),nbf.v4.new_code_cell('pd.read_csv("outputs/dispatch_strategy_kpis.csv")'),nbf.v4.new_code_cell('pd.read_csv("outputs/sample_new_job_rankings.csv").head(10)')]; nbf.write(nb,BASE/'technician_job_matching_demo.ipynb')
except Exception as e: print('notebook skipped',e)

ZIP=Path('/mnt/data/technician_job_matching_project.zip')
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in BASE.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(BASE.parent))
print('\nMODEL METRICS\n',metrics_df.to_string(index=False))
print('\nDISPATCH KPIS\n',kpis.to_string(index=False))
print('\nBEST',best_name,'TOP1',round(top1,4),'TOP3',round(top3,4))
print('\nSAMPLE TOP5\n',newdf[['rank','technician_id','predicted_suitability_score','relevant_skill','inventory_match','distance_km','available_in_minutes']].head().to_string(index=False))
print('\nZIP',ZIP)
