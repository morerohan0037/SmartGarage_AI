from pathlib import Path
import warnings, math, json, zipfile
import numpy as np, pandas as pd
BASE=Path('/mnt/data/technician_job_matching_project'); SEED=42
pairs=pd.read_csv(BASE/'data'/'technician_job_candidates_seed42.csv'); jobs=pd.read_csv(BASE/'data'/'jobs_seed42.csv'); techs=pd.read_csv(BASE/'data'/'technicians_seed42.csv')
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

ids=jobs.job_id.to_numpy(); train_ids,test_ids=train_test_split(ids,test_size=.20,random_state=SEED)
features=['job_branch','technician_branch','service_type','problem_type','urgency','required_part','equipment_age_years','previous_repairs','estimated_duration_minutes','required_skill','experience_years','relevant_skill','customer_rating','historical_first_fix_rate','callback_rate','current_workload_ratio','available_in_minutes','distance_km','travel_time_minutes','inventory_match','same_branch','skill_gap','skill_match_ratio']
cat=['job_branch','technician_branch','service_type','problem_type','urgency','required_part']; num=[c for c in features if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('ohe',OneHotEncoder(handle_unknown='ignore'))]),cat)])
models={
 'Linear Regression':LinearRegression(),
 'Decision Tree':DecisionTreeRegressor(max_depth=9,min_samples_leaf=12,random_state=SEED),
 'Random Forest':RandomForestRegressor(n_estimators=80,max_depth=13,min_samples_leaf=4,random_state=SEED,n_jobs=2),
 'XGBoost':XGBRegressor(n_estimators=100,max_depth=4,learning_rate=.06,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',random_state=SEED,n_jobs=2),
 'CatBoost':CatBoostRegressor(iterations=100,depth=6,learning_rate=.07,loss_function='RMSE',random_seed=SEED,verbose=False,thread_count=2),
 'Neural Network':MLPRegressor(hidden_layer_sizes=(32,16),max_iter=120,early_stopping=True,random_state=SEED)
}
train=pairs[pairs.job_id.isin(train_ids)]; test=pairs[pairs.job_id.isin(test_ids)]
# Fit preprocessing once. All regressors, including CatBoost, receive the same numeric feature matrix.
Xtr=pre.fit_transform(train[features]); Xte=pre.transform(test[features]); ytr=train.true_suitability_score.to_numpy(); yte=test.true_suitability_score.to_numpy()
joblib.dump(pre,BASE/'models'/'preprocessor.joblib')
metrics=[]; fitted={}; preds={}
for name,m in models.items():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); m.fit(Xtr,ytr)
    pr=np.clip(m.predict(Xte),0,100); fitted[name]=m; preds[name]=pr
    metrics.append({'Model':name,'MAE':mean_absolute_error(yte,pr),'RMSE':mean_squared_error(yte,pr)**.5,'R2':r2_score(yte,pr)})
    if name=='CatBoost': m.save_model(str(BASE/'models'/'catboost.cbm'))
    else: joblib.dump(m,BASE/'models'/(name.lower().replace(' ','_')+'.joblib'))
metrics_df=pd.DataFrame(metrics).sort_values(['MAE','RMSE']).reset_index(drop=True); metrics_df.to_csv(BASE/'outputs'/'model_metrics.csv',index=False)
best_name=metrics_df.iloc[0].Model; best=fitted[best_name]; scored=test.copy(); scored['predicted_suitability_score']=np.round(preds[best_name],3); scored.to_csv(BASE/'outputs'/'test_candidate_predictions.csv',index=False)
# strategies
ml=scored.loc[scored.groupby('job_id').predicted_suitability_score.idxmax()].copy(); oracle=scored.loc[scored.groupby('job_id').true_suitability_score.idxmax()].copy(); nearest=scored.loc[scored.groupby('job_id').sim_response_time_minutes.idxmin()].copy()
simple=scored.copy(); simple['rule_score']=30*np.clip(simple.skill_match_ratio,0,1.25)/1.25+25*simple.inventory_match+12*simple.historical_first_fix_rate+10*(simple.customer_rating/5)+8*simple.same_branch+8*(1-simple.current_workload_ratio)+7*(1-np.clip(simple.distance_km/80,0,1)); rule=simple.loc[simple.groupby('job_id').rule_score.idxmax()].copy(); rand=scored.groupby('job_id',group_keys=False).sample(n=1,random_state=SEED)
oracle_map=oracle.set_index('job_id').true_suitability_score
def kpi(name,df):
 reg=oracle_map.loc[df.job_id].to_numpy()-df.true_suitability_score.to_numpy()
 return {'Strategy':name,'Avg True Suitability':df.true_suitability_score.mean(),'First-Time Fix Rate':df.sim_first_time_fix.mean(),'SLA Met Rate':df.sim_sla_met.mean(),'Avg Response Time (min)':df.sim_response_time_minutes.mean(),'Avg Customer Satisfaction':df.sim_customer_satisfaction.mean(),'Avg Regret vs Oracle':reg.mean()}
kpis=pd.DataFrame([kpi('Random Assignment',rand),kpi('Nearest/Available',nearest),kpi('Weighted Rule',rule),kpi(f'ML Ranking ({best_name})',ml),kpi('Oracle Upper Bound',oracle)])
kpis.to_csv(BASE/'outputs'/'dispatch_strategy_kpis.csv',index=False); ml.to_csv(BASE/'outputs'/'ml_selected_technicians.csv',index=False)
oracle_best=oracle.set_index('job_id').technician_id; ml_best=ml.set_index('job_id').technician_id; top1=float((oracle_best.loc[ml_best.index]==ml_best).mean()); top3=float(np.mean([oracle_best.loc[j] in set(g.nlargest(3,'predicted_suitability_score').technician_id) for j,g in scored.groupby('job_id')]))
# sample new job rank all technicians
new={'job_branch':'East','service_type':'Automatic Gate Repair','problem_type':'Gate Motor Failure','urgency':'Emergency','equipment_age_years':12.0,'previous_repairs':3,'estimated_duration_minutes':125,'required_skill':5,'required_part':'motor','job_lat':25.0,'job_lon':28.0}
r=[]
for _,t in techs.iterrows():
 rel=t.gate_electrical_skill; dist=math.hypot(new['job_lat']-t.tech_lat,new['job_lon']-t.tech_lon)*1.35; travel=dist/42*60*1.25; inv=int(t.has_motor); same=int(t.branch==new['job_branch'])
 r.append({'job_branch':new['job_branch'],'technician_branch':t.branch,'service_type':new['service_type'],'problem_type':new['problem_type'],'urgency':new['urgency'],'required_part':new['required_part'],'equipment_age_years':new['equipment_age_years'],'previous_repairs':new['previous_repairs'],'estimated_duration_minutes':new['estimated_duration_minutes'],'required_skill':new['required_skill'],'experience_years':t.experience_years,'relevant_skill':rel,'customer_rating':t.customer_rating,'historical_first_fix_rate':t.historical_first_fix_rate,'callback_rate':t.callback_rate,'current_workload_ratio':t.current_workload_ratio,'available_in_minutes':t.available_in_minutes,'distance_km':round(dist,2),'travel_time_minutes':round(travel,1),'inventory_match':inv,'same_branch':same,'skill_gap':float(rel-new['required_skill']),'skill_match_ratio':round(rel/new['required_skill'],3),'technician_id':t.technician_id})
newdf=pd.DataFrame(r); newdf['predicted_suitability_score']=np.clip(best.predict(pre.transform(newdf[features])),0,100); newdf=newdf.sort_values('predicted_suitability_score',ascending=False).reset_index(drop=True); newdf['rank']=np.arange(1,len(newdf)+1); newdf.to_csv(BASE/'outputs'/'sample_new_job_rankings.csv',index=False)
# feature importance
try:
 names=pre.get_feature_names_out(); model=best; vals=model.feature_importances_ if hasattr(model,'feature_importances_') else np.abs(np.asarray(model.coef_).ravel()) if hasattr(model,'coef_') else None
 if vals is not None and len(vals)==len(names): pd.DataFrame({'feature':names,'importance':vals}).sort_values('importance',ascending=False).to_csv(BASE/'outputs'/'feature_importance.csv',index=False)
except Exception: pass
# plots
import matplotlib.pyplot as plt
p=metrics_df.sort_values('MAE'); plt.figure(figsize=(9,5)); plt.barh(p.Model,p.MAE); plt.xlabel('MAE (lower is better)'); plt.title('Technician Suitability Model Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'model_comparison_mae.png',dpi=180); plt.close()
p=metrics_df.sort_values('R2'); plt.figure(figsize=(9,5)); plt.barh(p.Model,p.R2); plt.xlabel('R² (higher is better)'); plt.title('Model R² Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'model_comparison_r2.png',dpi=180); plt.close()
p=kpis.sort_values('Avg True Suitability'); plt.figure(figsize=(10,5)); plt.barh(p.Strategy,p['Avg True Suitability']); plt.xlabel('Average true suitability'); plt.title('Dispatch Strategy Comparison'); plt.tight_layout(); plt.savefig(BASE/'plots'/'dispatch_strategy_suitability.png',dpi=180); plt.close()
p=kpis.sort_values('First-Time Fix Rate'); plt.figure(figsize=(10,5)); plt.barh(p.Strategy,p['First-Time Fix Rate']*100); plt.xlabel('First-time fix rate (%)'); plt.title('First-Time Fix by Dispatch Strategy'); plt.tight_layout(); plt.savefig(BASE/'plots'/'dispatch_strategy_first_fix.png',dpi=180); plt.close()
s=scored.sample(min(1800,len(scored)),random_state=SEED); plt.figure(figsize=(6.5,6)); plt.scatter(s.true_suitability_score,s.predicted_suitability_score,alpha=.25,s=12); plt.plot([0,100],[0,100],'--'); plt.xlabel('True suitability'); plt.ylabel('Predicted suitability'); plt.title(f'Actual vs Predicted ({best_name})'); plt.tight_layout(); plt.savefig(BASE/'plots'/'actual_vs_predicted.png',dpi=180); plt.close()
top=newdf.head(10).sort_values('predicted_suitability_score'); plt.figure(figsize=(9,5)); plt.barh(top.technician_id,top.predicted_suitability_score); plt.xlabel('Predicted suitability'); plt.title('Top 10 Technicians for Sample Emergency Gate Job'); plt.tight_layout(); plt.savefig(BASE/'plots'/'sample_job_top10.png',dpi=180); plt.close()
# dictionary & helper files
pd.DataFrame([('job_id','Unique service job ID'),('technician_id','Candidate technician ID'),('service_type','Garage/gate service category'),('problem_type','Specific service problem'),('urgency','Low, Normal, High, Emergency'),('required_skill','Required skill level 1-5'),('required_part','Likely needed part'),('experience_years','Technician experience'),('relevant_skill','Technician skill relevant to job 1-5'),('historical_first_fix_rate','Technician historical one-visit success rate'),('callback_rate','Technician historical repeat-visit rate'),('current_workload_ratio','Current workload 0-1'),('available_in_minutes','Minutes until available'),('distance_km','Technician-to-job distance'),('travel_time_minutes','Estimated travel time'),('inventory_match','1 if required part is carried'),('same_branch','1 if same branch'),('skill_gap','Relevant skill minus required skill'),('skill_match_ratio','Relevant skill / required skill'),('true_suitability_score','Synthetic target 0-100'),('sim_first_time_fix','Simulated first-time-fix outcome'),('sim_sla_met','Simulated SLA outcome'),('sim_response_time_minutes','Availability + travel'),('sim_customer_satisfaction','Simulated satisfaction 1-5')],columns=['column','description']).to_csv(BASE/'data'/'data_dictionary.csv',index=False)
(BASE/'requirements.txt').write_text('numpy\npandas\nscikit-learn\nmatplotlib\njoblib\nxgboost\ncatboost\njupyter\n')
(BASE/'predict_new_job.py').write_text('from pathlib import Path\nimport pandas as pd\nB=Path(__file__).resolve().parent\ndf=pd.read_csv(B/"outputs"/"sample_new_job_rankings.csv")\nprint(df[["rank","technician_id","predicted_suitability_score","relevant_skill","inventory_match","distance_km","available_in_minutes"]].head(10).to_string(index=False))\n')
bestrow=metrics_df.iloc[0]; mlk=kpis[kpis.Strategy.str.startswith('ML Ranking')].iloc[0]; nk=kpis[kpis.Strategy=='Nearest/Available'].iloc[0]; rk=kpis[kpis.Strategy=='Weighted Rule'].iloc[0]
(BASE/'README.md').write_text(f'''# Technician-Job Matching / Suitability Score\n\nReproducible synthetic garage-door and automatic-gate field-service ML project. Seed = 42.\n\n## Dataset\n2500 jobs, 40 technicians, 8 candidates/job, {len(pairs):,} candidate pairs.\n\n## Models\nLinear Regression, Decision Tree, Random Forest, XGBoost, CatBoost, Neural Network.\n\n## Best model\n{best_name}: MAE={bestrow.MAE:.3f}, RMSE={bestrow.RMSE:.3f}, R²={bestrow.R2:.3f}.\n\nTop-1 oracle agreement: {top1*100:.1f}%\nTop-3 oracle capture: {top3*100:.1f}%\n\n## Operational comparison\nNearest/Available: suitability={nk['Avg True Suitability']:.2f}, FTF={nk['First-Time Fix Rate']*100:.1f}%, SLA={nk['SLA Met Rate']*100:.1f}%, response={nk['Avg Response Time (min)']:.1f} min.\nWeighted Rule: suitability={rk['Avg True Suitability']:.2f}, FTF={rk['First-Time Fix Rate']*100:.1f}%, SLA={rk['SLA Met Rate']*100:.1f}%, response={rk['Avg Response Time (min)']:.1f} min.\nML Ranking: suitability={mlk['Avg True Suitability']:.2f}, FTF={mlk['First-Time Fix Rate']*100:.1f}%, SLA={mlk['SLA Met Rate']*100:.1f}%, response={mlk['Avg Response Time (min)']:.1f} min.\n\nImportant: synthetic prototype data, not real Precision Garage Door Service records.\n''')
(BASE/'outputs'/'run_summary.json').write_text(json.dumps({'seed':42,'best_model':best_name,'top1_oracle_match':top1,'top3_oracle_capture':top3,'best_model_metrics':{'mae':float(bestrow.MAE),'rmse':float(bestrow.RMSE),'r2':float(bestrow.R2)}},indent=2))
# simple notebook
try:
 import nbformat as nbf
 nb=nbf.v4.new_notebook(); nb.cells=[nbf.v4.new_markdown_cell('# Technician-Job Matching ML Project\nSynthetic data, seed 42.'),nbf.v4.new_code_cell('import pandas as pd\npairs=pd.read_csv("data/technician_job_candidates_seed42.csv")\npairs.head()'),nbf.v4.new_code_cell('pd.read_csv("outputs/model_metrics.csv")'),nbf.v4.new_code_cell('pd.read_csv("outputs/dispatch_strategy_kpis.csv")'),nbf.v4.new_code_cell('pd.read_csv("outputs/sample_new_job_rankings.csv").head(10)')]; nbf.write(nb,BASE/'technician_job_matching_demo.ipynb')
except Exception: pass
ZIP=Path('/mnt/data/technician_job_matching_project.zip')
if ZIP.exists(): ZIP.unlink()
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
    for p in BASE.rglob('*'):
        if p.is_file(): z.write(p,p.relative_to(BASE.parent))
print('METRICS'); print(metrics_df.to_string(index=False)); print('\nKPIS'); print(kpis.to_string(index=False)); print('\nBEST',best_name,'TOP1',top1,'TOP3',top3); print('\nTOP5'); print(newdf[['rank','technician_id','predicted_suitability_score','relevant_skill','inventory_match','distance_km','available_in_minutes']].head().to_string(index=False)); print('\nZIP',ZIP)
