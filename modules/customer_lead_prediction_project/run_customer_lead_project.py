"""Reproducible garage-door / automatic-gate lead conversion project.
Usage: python run_customer_lead_project.py
"""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
SEED=42
ROOT=Path(__file__).resolve().parent
for p in ['data','models','outputs','plots']:(ROOT/p).mkdir(exist_ok=True)
FEATURES=['branch','service_type','lead_source','property_type','customer_type','urgency','contact_channel','quote_requested',
          'door_age_years','previous_repairs','prior_customer','response_time_minutes','quote_amount_usd','budget_usd',
          'distance_km','competitor_quotes','warranty_interest','financing_interest','appointment_slots_next_3days',
          'sales_rep_experience_years','month','day_of_week','is_weekend','price_to_budget_ratio','available_slot_indicator']
CATS=['branch','service_type','lead_source','property_type','customer_type','urgency','contact_channel']
NUMS=[x for x in FEATURES if x not in CATS]

def generate_data(n=16000,seed=SEED):
    r=np.random.default_rng(seed)
    start=np.datetime64('2024-01-01'); days=r.integers(0, 1000, n)
    dates=pd.to_datetime(start+days.astype('timedelta64[D]'))
    services=r.choice(['Repair','Opener Replacement','New Garage Door','Automatic Gate Repair','Gate Installation','Maintenance'],n,
                      p=[.34,.18,.17,.16,.08,.07])
    src=r.choice(['Organic Search','Paid Search','Referral','Website Direct','Social Media','Existing Customer'],n,
                 p=[.27,.23,.18,.14,.09,.09])
    age=np.round(r.gamma(2.4,3.7,n).clip(0,28),1)
    urgent=r.choice(['Low','Normal','High','Emergency'],n,p=[.17,.49,.24,.10])
    typical={'Repair':275,'Opener Replacement':550,'New Garage Door':2150,'Automatic Gate Repair':475,
             'Gate Installation':4100,'Maintenance':165}
    quote=np.round(np.array([typical[x] for x in services])*r.lognormal(0,.25,n),2)
    budgets=np.round(quote*r.lognormal(.08,.37,n),2)
    response=np.round(r.gamma(2,34,n).clip(1,300),1)
    slots=r.integers(0,7,n)
    prev=r.poisson(np.clip(age/9,0,4)).clip(0,8)
    prior=r.binomial(1,.22,n)
    channel=r.choice(['Phone','Web Form','Chat','Email'],n,p=[.48,.31,.13,.08])
    branch=r.choice(['North','South','East','West','Central'],n)
    dist=np.round(r.gamma(2,5,n).clip(.5,45),1)
    comp=r.poisson(1,n).clip(0,5)
    warranty=r.binomial(1,.38,n)
    financing=r.binomial(1,.23,n)
    quote_requested=r.binomial(1,.73,n)
    rep=np.round(r.gamma(2,2.2,n).clip(.2,15),1)
    prop=r.choice(['Single Family','Townhouse','Commercial','Multi Unit'],n,p=[.67,.15,.14,.04])
    cust=r.choice(['Residential','Commercial'],n,p=[.82,.18])
    ratio=np.clip(quote/budgets,.25,4)
    # Latent conversion mechanism is synthetic, with nonlinear interactions and stochastic outcomes.
    z=(-1.75 + .70*(src=='Referral')+.51*(src=='Existing Customer')+.28*(src=='Organic Search')
       -.34*(src=='Social Media') +.65*prior+.44*quote_requested
       +.32*(urgent=='Emergency')+.22*(urgent=='High')-.19*(urgent=='Low')
       -.008*response-.55*np.maximum(ratio-1,0)+.26*(ratio<.90)
       -.22*comp+.12*warranty+.10*(financing & (quote>1100))
       +.12*np.minimum(rep,10)/5+.21*(slots>=2)-.30*(slots==0)
       -.008*dist+.09*(age>=13)+.07*np.minimum(prev,4)
       +.22*((services=='New Garage Door')&(age>15))
       -.30*((services=='Gate Installation')&(ratio>1.2))
       +.10*np.sin(2*np.pi*dates.month.to_numpy()/12)
       +r.normal(0,.45,n))
    probability=1/(1+np.exp(-z))
    converted=r.binomial(1,probability)
    df=pd.DataFrame({'lead_id':[f'L{i:06d}' for i in range(1,n+1)],'created_date':dates,
      'branch':branch,'service_type':services,'lead_source':src,'property_type':prop,'customer_type':cust,
      'urgency':urgent,'contact_channel':channel,'quote_requested':quote_requested,
      'door_age_years':age,'previous_repairs':prev,'prior_customer':prior,'response_time_minutes':response,
      'quote_amount_usd':quote,'budget_usd':budgets,'distance_km':dist,'competitor_quotes':comp,
      'warranty_interest':warranty,'financing_interest':financing,'appointment_slots_next_3days':slots,
      'sales_rep_experience_years':rep,'month':dates.month,'day_of_week':dates.dayofweek,
      'is_weekend':(dates.dayofweek>=5).astype(int),'price_to_budget_ratio':np.round(ratio,3),
      'available_slot_indicator':(slots>0).astype(int),'converted':converted})
    return df.sort_values(['created_date','lead_id']).reset_index(drop=True)

def make_prep():
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUMS),
                              ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                                                    ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CATS)])

def models():
    m={'Logistic Regression':LogisticRegression(max_iter=1000,random_state=SEED),
       'Decision Tree':DecisionTreeClassifier(max_depth=7,min_samples_leaf=35,random_state=SEED),
       'Random Forest':RandomForestClassifier(n_estimators=120,max_depth=12,min_samples_leaf=7,n_jobs=-1,random_state=SEED),
       'Hist Gradient Boosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=19,learning_rate=.045,random_state=SEED),
       'Neural Network':MLPClassifier(hidden_layer_sizes=(40,20),max_iter=160,early_stopping=True,random_state=SEED)}
    try:
        from xgboost import XGBClassifier
        m['XGBoost']=XGBClassifier(n_estimators=160,max_depth=3,learning_rate=.045,subsample=.88,colsample_bytree=.9,
                                  eval_metric='logloss',random_state=SEED,n_jobs=2)
    except ImportError: print('XGBoost unavailable, skipping.')
    return m

def score(y,proba,threshold):
    pred=(proba>=threshold).astype(int)
    return {'Accuracy':accuracy_score(y,pred),'Precision':precision_score(y,pred,zero_division=0),
            'Recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),
            'ROC_AUC':roc_auc_score(y,proba),'PR_AUC':average_precision_score(y,proba),
            'Brier':brier_score_loss(y,proba)}

def main():
    df=generate_data(); df.to_csv(ROOT/'data/leads_synthetic_seed42.csv',index=False)
    # Chronological split, avoiding overly optimistic random split across time.
    n=len(df); train=df.iloc[:int(n*.65)]; val=df.iloc[int(n*.65):int(n*.80)]; test=df.iloc[int(n*.80):]
    yval=val.converted.to_numpy(); ytest=test.converted.to_numpy()
    stats=[]; fitted={}; curves={}; thresholds={}
    for name, estimator in models().items():
        print('Training',name,flush=True)
        pipe=Pipeline([('prep',make_prep()),('model',estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');pipe.fit(train[FEATURES],train.converted)
        vp=pipe.predict_proba(val[FEATURES])[:,1]
        grid=np.arange(.20,.81,.025)
        f1values=[f1_score(yval,vp>=t,zero_division=0) for t in grid]
        threshold=float(grid[np.argmax(f1values)])
        tp=pipe.predict_proba(test[FEATURES])[:,1]
        stats.append({'Model':name,'Threshold':threshold,**score(ytest,tp,threshold)})
        fitted[name]=pipe;curves[name]=tp;thresholds[name]=threshold
        joblib.dump(pipe,ROOT/'models'/f'{name.lower().replace(" ","_")}.joblib')
    met=pd.DataFrame(stats).sort_values('PR_AUC',ascending=False).reset_index(drop=True)
    met.to_csv(ROOT/'outputs/model_metrics.csv',index=False)
    best=met.iloc[0].Model; bp=curves[best];th=thresholds[best]
    testout=test[['lead_id','created_date','converted','service_type','lead_source']].copy()
    testout['conversion_probability']=np.round(bp,5)
    testout['predicted_conversion']=(bp>=th).astype(int)
    testout['lead_priority']=pd.cut(bp,[-.01,.2,.5,.75,1.01],labels=['Low','Medium','High','Very High'])
    testout.to_csv(ROOT/'outputs/test_lead_predictions.csv',index=False)
    # Non-ML baseline: prioritize referrals and existing customers; 0/1 threshold predictions.
    rule=(test['lead_source'].isin(['Referral','Existing Customer']) | (test['prior_customer']==1)).astype(int).to_numpy()
    baseline=score(ytest,rule,.5)
    pd.DataFrame([{'Approach':'Source and loyalty rule',**baseline},
                  {'Approach':f'ML: {best}',**score(ytest,bp,th)}]).to_csv(ROOT/'outputs/baseline_comparison.csv',index=False)
    # Top 20% conversion targeting: measure yield and lift.
    k=max(1,int(len(test)*.2)); top=np.argsort(-bp)[:k]
    random_rate=ytest.mean();top_rate=ytest[top].mean();lift=top_rate/random_rate
    # Business illustration: identical hypothetical $20 outreach cost, $160 gross contribution per conversion
    cost_per_contact=20; margin_per_conversion=160
    targeting=pd.DataFrame([{'Strategy':'Contact all leads','Leads Contacted':len(test),
                              'Expected Campaign Net Contribution USD':int(ytest.sum())*margin_per_conversion-len(test)*cost_per_contact,
                              'Observed Conversion Rate':random_rate},
                             {'Strategy':'Contact top 20% by ML score','Leads Contacted':k,
                              'Expected Campaign Net Contribution USD':int(ytest[top].sum())*margin_per_conversion-k*cost_per_contact,
                              'Observed Conversion Rate':top_rate}])
    targeting.to_csv(ROOT/'outputs/targeting_business_scenario.csv',index=False)
    # New lead demonstration: alter actionable pre-sale features from held-out examples.
    examples=test.iloc[[3,40,80,140]].copy().drop(columns='converted')
    examples['conversion_probability']=np.round(fitted[best].predict_proba(examples[FEATURES])[:,1],4)
    examples['predicted_conversion']=(examples.conversion_probability>=th).astype(int)
    examples.to_csv(ROOT/'outputs/sample_new_lead_predictions.csv',index=False)
    # plots
    plt.figure(figsize=(9,5)); ordered=met.sort_values('PR_AUC');plt.barh(ordered.Model,ordered.PR_AUC)
    plt.title('Model Comparison: PR-AUC');plt.xlabel('Average precision (higher is better)');plt.tight_layout();plt.savefig(ROOT/'plots/model_comparison_prauc.png',dpi=150);plt.close()
    plt.figure(figsize=(8,5))
    for name,p in curves.items():
        fpr,tpr,_=roc_curve(ytest,p);plt.plot(fpr,tpr,label=name)
    plt.plot([0,1],[0,1],linestyle=':');plt.legend(fontsize=7);plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title('ROC Curves');plt.tight_layout();plt.savefig(ROOT/'plots/roc_curves.png',dpi=150);plt.close()
    plt.figure(figsize=(8,5))
    for name,p in curves.items():
        precision,recall,_=precision_recall_curve(ytest,p);plt.plot(recall,precision,label=name)
    plt.axhline(random_rate,linestyle=':',label='Base conversion rate');plt.legend(fontsize=7)
    plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Precision Recall Curves');plt.tight_layout();plt.savefig(ROOT/'plots/precision_recall_curves.png',dpi=150);plt.close()
    cm=confusion_matrix(ytest,bp>=th)
    plt.figure(figsize=(5,4));plt.imshow(cm,cmap='Blues');plt.xticks([0,1],['Not converted','Converted']);plt.yticks([0,1],['Not converted','Converted']);
    for i in range(2):
        for j in range(2):plt.text(j,i,str(cm[i,j]),ha='center',va='center')
    plt.xlabel('Predicted');plt.ylabel('Actual');plt.title('Best Model Confusion Matrix');plt.tight_layout();plt.savefig(ROOT/'plots/confusion_matrix.png',dpi=150);plt.close()
    # Lift chart deciles
    sortidx=np.argsort(-bp); deciles=[]
    for j in range(10):
        ix=sortidx[j*len(test)//10:(j+1)*len(test)//10]
        deciles.append({'Decile':j+1,'Conversion Rate':float(ytest[ix].mean()),'Lift':float(ytest[ix].mean()/random_rate)})
    pd.DataFrame(deciles).to_csv(ROOT/'outputs/lift_by_decile.csv',index=False)
    plt.figure(figsize=(8,5));plt.bar(range(1,11),[d['Lift'] for d in deciles]);plt.axhline(1,linestyle=':');plt.xlabel('Predicted score decile (1 = highest)');plt.ylabel('Lift over average');plt.title('Lead Ranking Lift');plt.tight_layout();plt.savefig(ROOT/'plots/lift_by_decile.png',dpi=150);plt.close()
    group=df.groupby('lead_source').converted.mean().sort_values()
    plt.figure(figsize=(9,5));plt.barh(group.index,group.values*100);plt.xlabel('Synthetic conversion rate (%)');plt.title('Lead Conversion by Source');plt.tight_layout();plt.savefig(ROOT/'plots/conversion_by_source.png',dpi=150);plt.close()
    # Permutation importance on held-out data, small set, to work with all model types
    from sklearn.inspection import permutation_importance
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        imp=permutation_importance(fitted[best],test[FEATURES].sample(n=min(700,len(test)),random_state=SEED),
          test.loc[test[FEATURES].sample(n=min(700,len(test)),random_state=SEED).index,'converted'],
          n_repeats=3,scoring='average_precision',random_state=SEED,n_jobs=1)
    importance=pd.DataFrame({'Feature':FEATURES,'Importance':imp.importances_mean}).sort_values('Importance',ascending=False)
    importance.to_csv(ROOT/'outputs/feature_importance.csv',index=False)
    topf=importance.head(12).sort_values('Importance');plt.figure(figsize=(9,6));plt.barh(topf.Feature,topf.Importance)
    plt.title('Permutation Feature Importance');plt.xlabel('Mean decrease in average precision');plt.tight_layout();plt.savefig(ROOT/'plots/feature_importance.png',dpi=150);plt.close()
    summary={'seed':SEED,'rows':len(df),'train_rows':len(train),'validation_rows':len(val),'test_rows':len(test),
      'conversion_rate':float(df.converted.mean()),'best_model_by_pr_auc':best,
      'threshold_selected_on_validation':th,'test_metrics':{k:float(v) for k,v in score(ytest,bp,th).items()},
      'top_20pct_conversion_rate':float(top_rate),'overall_test_conversion_rate':float(random_rate),
      'top_20pct_lift':float(lift),'note':'Metrics are on synthetic data; targeting margin/cost assumptions are illustrative.'}
    (ROOT/'outputs/run_summary.json').write_text(json.dumps(summary,indent=2))
    print('\nMODEL RESULTS\n',met.to_string(index=False)); print('\nSummary:',json.dumps(summary,indent=2))
    print('\nNEW LEAD DEMO\n',examples[['lead_id','service_type','lead_source','conversion_probability','predicted_conversion']].to_string(index=False))
    print('\nSuccess: generated data, trained models, plots, predictions and outputs.')
if __name__=='__main__':main()
