from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,confusion_matrix,brier_score_loss)
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parent
for folder in ['models','outputs','plots']:(ROOT/folder).mkdir(exist_ok=True)
SEED=42
DATA=ROOT/'data/customer_churn_business_dataset.csv'
df=pd.read_csv(DATA)
assert set(df.churn.unique()).issubset({0,1})
# Avoid identity leakage and customer demographics unsuitable for retention treatment decisions.
exclude=['customer_id','churn','gender','age','country','city']
features=[c for c in df.columns if c not in exclude]
cat=[c for c in features if df[c].dtype=='object']
num=[c for c in features if c not in cat]
trv,te=train_test_split(df,test_size=.2,random_state=SEED,stratify=df.churn)
tr,va=train_test_split(trv,test_size=.25,random_state=SEED,stratify=trv.churn)
def preprocess():
 return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
models={
 'Logistic Regression':LogisticRegression(max_iter=1200,random_state=SEED,class_weight='balanced'),
 'Decision Tree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=30,random_state=SEED,class_weight='balanced'),
 'Random Forest':RandomForestClassifier(n_estimators=140,max_depth=12,min_samples_leaf=5,random_state=SEED,class_weight='balanced',n_jobs=-1),
 'Gradient Boosting':GradientBoostingClassifier(n_estimators=130,learning_rate=.05,max_depth=2,random_state=SEED),
 'Hist Gradient Boosting':HistGradientBoostingClassifier(max_iter=130,max_leaf_nodes=16,learning_rate=.05,random_state=SEED),
 'Neural Network':MLPClassifier(hidden_layer_sizes=(40,20),max_iter=180,early_stopping=True,random_state=SEED)
}
try:
 from xgboost import XGBClassifier
 models['XGBoost']=XGBClassifier(n_estimators=170,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=SEED,n_jobs=2)
except ImportError:print('XGBoost unavailable; skipping')
# CatBoost is omitted because its installed sklearn wrapper is incompatible with the current environment.
results=[]; fitted={}; validation=[]; roc_data={}; pr_data={}
for name,est in models.items():
 print('Training:',name,flush=True)
 pipe=Pipeline([('preprocess',preprocess()),('model',est)])
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');pipe.fit(tr[features],tr.churn)
 vp=pipe.predict_proba(va[features])[:,1]
 grid=np.arange(.15,.851,.025)
 scores=[f1_score(va.churn,(vp>=t).astype(int),zero_division=0) for t in grid]
 threshold=float(grid[int(np.argmax(scores))])
 prob=pipe.predict_proba(te[features])[:,1]; pred=(prob>=threshold).astype(int)
 result={'Model':name,'Validation_PR_AUC':average_precision_score(va.churn,vp),'Threshold':threshold,'Accuracy':accuracy_score(te.churn,pred),'Precision':precision_score(te.churn,pred,zero_division=0),'Recall':recall_score(te.churn,pred,zero_division=0),'F1':f1_score(te.churn,pred,zero_division=0),'ROC_AUC':roc_auc_score(te.churn,prob),'PR_AUC':average_precision_score(te.churn,prob),'Brier':brier_score_loss(te.churn,prob)}
 results.append(result);fitted[name]=pipe
 validation.append({'Model':name,'Selected threshold':threshold,'Validation F1':max(scores)})
 roc_data[name]=roc_curve(te.churn,prob)[:2];pr_data[name]=precision_recall_curve(te.churn,prob)[:2]
 joblib.dump(pipe,ROOT/'models'/f'{name.lower().replace(" ","_")}.joblib')
metrics=pd.DataFrame(results).sort_values(['Validation_PR_AUC','ROC_AUC'],ascending=False).reset_index(drop=True)
metrics.to_csv(ROOT/'outputs/model_metrics.csv',index=False)
pd.DataFrame(validation).to_csv(ROOT/'outputs/validation_thresholds.csv',index=False)
best_name=metrics.iloc[0]['Model'];best=fitted[best_name];thr=float(metrics.iloc[0]['Threshold'])
prob=best.predict_proba(te[features])[:,1]; pred=(prob>=thr).astype(int)
predictions=te[['customer_id','churn','customer_segment','contract_type','monthly_fee','tenure_months']].copy()
predictions['churn_probability']=prob;predictions['predicted_churn']=pred
predictions['risk_band']=pd.cut(prob,bins=[-0.001,.3,.6,1.001],labels=['Low','Medium','High'])
predictions.sort_values('churn_probability',ascending=False).to_csv(ROOT/'outputs/test_set_predictions.csv',index=False)
# Retention prioritization: pure ranking performance on untouched test set.
ordered=predictions.sort_values('churn_probability',ascending=False).reset_index(drop=True)
rows=[]
for fraction in [.05,.10,.20,.30,.50,1.0]:
 k=max(1,round(len(ordered)*fraction));picked=ordered.head(k)
 rows.append({'Top fraction':fraction,'Leads targeted':k,'Churn rate in segment':picked.churn.mean(),'Recall of churners':picked.churn.sum()/max(1,ordered.churn.sum()),'Lift over test base rate':picked.churn.mean()/ordered.churn.mean()})
pd.DataFrame(rows).to_csv(ROOT/'outputs/retention_top_k_lift.csv',index=False)
# A simple no-ML baseline using customer support events, not a real operational benchmark
baseline=((te.support_tickets>=3)|(te.payment_failures>=2)).astype(int)
base=pd.DataFrame([{'Strategy':'Majority class (no churn)','Accuracy':accuracy_score(te.churn,np.zeros(len(te))),'Precision':0,'Recall':0,'F1':0},{'Strategy':'Rule: tickets>=3 or payments>=2','Accuracy':accuracy_score(te.churn,baseline),'Precision':precision_score(te.churn,baseline,zero_division=0),'Recall':recall_score(te.churn,baseline,zero_division=0),'F1':f1_score(te.churn,baseline,zero_division=0)},{'Strategy':'ML: '+best_name,'Accuracy':accuracy_score(te.churn,pred),'Precision':precision_score(te.churn,pred,zero_division=0),'Recall':recall_score(te.churn,pred,zero_division=0),'F1':f1_score(te.churn,pred,zero_division=0)}])
base.to_csv(ROOT/'outputs/baseline_comparison.csv',index=False)
# Features should be available prior to churn measurement. Real snapshot timestamps are absent.
# Importance from heldout data (display only; NOT used for model selection).
imp=permutation_importance(best,te[features],te.churn,n_repeats=3,random_state=SEED,scoring='average_precision',n_jobs=1)
importance=pd.DataFrame({'feature':features,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False)
importance.to_csv(ROOT/'outputs/feature_importance.csv',index=False)
# New cases: use two actual test records but mask their outcome and identity from input
samples=te.iloc[:5][features].copy()
sp=best.predict_proba(samples)[:,1]
new=samples.copy();new.insert(0,'case_id',[f'DEMO_{i+1}' for i in range(len(samples))]);new['predicted_churn_probability']=sp;new['predicted_churn']=(sp>=thr).astype(int)
new.to_csv(ROOT/'outputs/sample_customer_predictions.csv',index=False)
plt.figure(figsize=(9,5));dd=metrics.sort_values('PR_AUC');plt.barh(dd.Model,dd.PR_AUC);plt.xlabel('Average precision / PR-AUC');plt.title('Churn Model Comparison');plt.tight_layout();plt.savefig(ROOT/'plots/model_comparison_pr_auc.png',dpi=170);plt.close()
plt.figure(figsize=(7,5))
for name,(fpr,tpr) in roc_data.items():plt.plot(fpr,tpr,label=name)
plt.plot([0,1],[0,1],'--',label='Chance');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title('ROC Curves');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(ROOT/'plots/roc_curves.png',dpi=170);plt.close()
plt.figure(figsize=(7,5))
for name,(precision,recall) in pr_data.items():plt.plot(recall,precision,label=name)
plt.axhline(te.churn.mean(),linestyle='--');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Precision-Recall Curves');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(ROOT/'plots/precision_recall_curves.png',dpi=170);plt.close()
cm=confusion_matrix(te.churn,pred);plt.figure(figsize=(5,4));plt.imshow(cm);plt.xticks([0,1],['Stay','Churn']);plt.yticks([0,1],['Stay','Churn']);plt.xlabel('Predicted');plt.ylabel('Actual');plt.title(f'Confusion Matrix: {best_name}')
for i in range(2):
 for j in range(2):plt.text(j,i,str(cm[i,j]),ha='center',va='center')
plt.tight_layout();plt.savefig(ROOT/'plots/confusion_matrix.png',dpi=170);plt.close()
plt.figure(figsize=(8,5));top=importance.head(12).sort_values('importance_mean');plt.barh(top.feature,top.importance_mean);plt.xlabel('Drop in test PR-AUC after permutation');plt.title('Feature Importance');plt.tight_layout();plt.savefig(ROOT/'plots/feature_importance.png',dpi=170);plt.close()
lf=pd.read_csv(ROOT/'outputs/retention_top_k_lift.csv');plt.figure(figsize=(7,5));plt.plot(lf['Top fraction']*100,lf['Lift over test base rate'],'o-');plt.axhline(1,linestyle='--');plt.xlabel('Top-ranked customers targeted (%)');plt.ylabel('Churn lift');plt.title('Retention Targeting Lift');plt.tight_layout();plt.savefig(ROOT/'plots/top_k_lift.png',dpi=170);plt.close()
plt.figure(figsize=(7,5));plt.hist(prob[te.churn.values==0],alpha=.6,bins=25,label='Retained');plt.hist(prob[te.churn.values==1],alpha=.6,bins=25,label='Churned');plt.xlabel('Predicted churn probability');plt.ylabel('Customers');plt.title('Churn Risk Distribution');plt.legend();plt.tight_layout();plt.savefig(ROOT/'plots/risk_distribution.png',dpi=170);plt.close()
summary={'dataset_rows':len(df),'dataset_columns':len(df.columns),'churn_rate':float(df.churn.mean()),'split':{'train':len(tr),'validation':len(va),'test':len(te)},'best_by_validation_pr_auc':best_name,'test_metrics':{k:float(metrics.iloc[0][k]) for k in ['Accuracy','Precision','Recall','F1','ROC_AUC','PR_AUC']},'selected_threshold':thr,'note':'Data supplied by user; not verified real company records. Observational predictive evaluation only.'}
(ROOT/'outputs/run_summary.json').write_text(json.dumps(summary,indent=2))
print('\n',metrics.to_string(index=False));print('\nTop-K:',lf.to_string(index=False));print('\nBaseline:',base.to_string(index=False));print('\nSaved files under',ROOT,flush=True)
