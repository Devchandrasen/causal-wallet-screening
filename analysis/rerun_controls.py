"""Rebuild sensitivities, controls and calibration from pinned raw inputs.

Horizon comparison uses the same first-appearance test range 37..42 for every
H, rather than claiming the inherited different-cohort figure was reproduced.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import reconstruct_wallet_experiments as reconstruction
from reconstruct_wallet_experiments import ACTIVITY, SNA, evaluate, select_f1_threshold
from audit_utils import environment, save_json, sha256, config_hash, utc_now


def xgb(y,seed=42):
    return XGBClassifier(n_estimators=350,max_depth=4,learning_rate=.04,subsample=.9,colsample_bytree=.9,
                         min_child_weight=2,reg_lambda=2,scale_pos_weight=float((y==0).sum()/(y==1).sum()),
                         random_state=seed,n_jobs=4,tree_method='hist')


def primary_controls(root):
    out=root/'controls';out.mkdir(exist_ok=True)
    samples=pd.read_parquet(root/'primary'/'wallet_samples.parquet')
    train=samples.split.eq('train');val=samples.split.eq('validation');test=samples.split.eq('test')
    asset=joblib.load(root/'primary'/'models'/'XGBoost_activity_sna_seed42.joblib')
    model=asset['model']
    importance=permutation_importance(model,samples.loc[val,ACTIVITY+SNA],samples.loc[val,'label'],scoring='average_precision',n_repeats=10,random_state=2026,n_jobs=1)
    rows=[]
    for i,feature in enumerate(ACTIVITY+SNA):
        for repeat,value in enumerate(importance.importances[i]):rows.append({'feature':feature,'repeat':repeat,'ap_decrease':value})
    pd.DataFrame(rows).to_csv(out/'permutation_importance.csv',index=False)
    rows=[];preds=[];vpreds=[]
    for seed in [1,2,3,4,5]:
        y=samples.loc[train,'label'].to_numpy()
        y=np.random.default_rng(seed).permutation(y)
        shuffled=xgb(y,seed)
        shuffled.fit(samples.loc[train,ACTIVITY+SNA],y)
        vs=shuffled.predict_proba(samples.loc[val,ACTIVITY+SNA])[:,1]
        ts=shuffled.predict_proba(samples.loc[test,ACTIVITY+SNA])[:,1]
        threshold=select_f1_threshold(samples.loc[val,'label'].to_numpy(),vs)
        rows.append({'seed':seed,**evaluate(samples.loc[test,'label'].to_numpy(),ts,threshold)})
        for mask,score,collection in [(val,vs,vpreds),(test,ts,preds)]:
            frame=samples.loc[mask,['address','label','first_step','cutoff']].copy()
            frame['score'],frame['threshold'],frame['seed']=score,threshold,seed
            collection.append(frame)
    pd.DataFrame(rows).to_csv(out/'label_permutation_metrics.csv',index=False)
    pd.concat(preds).to_parquet(out/'label_permutation_test_predictions.parquet',index=False)
    pd.concat(vpreds).to_parquet(out/'label_permutation_validation_predictions.parquet',index=False)
    v=pd.read_parquet(root/'primary'/'validation_predictions.parquet')
    t=pd.read_parquet(root/'primary'/'test_predictions.parquet')
    v=v[(v.learner=='XGBoost')&(v.seed==42)&(v.features=='activity_sna')]
    t=t[(t.learner=='XGBoost')&(t.seed==42)&(t.features=='activity_sna')]
    logit=lambda score:np.log(np.clip(score,1e-7,1-1e-7)/np.clip(1-score,1e-7,1-1e-7)).reshape(-1,1)
    platt=LogisticRegression(C=1e6,solver='lbfgs').fit(logit(v.score.to_numpy()),v.label)
    assert platt.coef_[0,0]>0,'Do not choose a test score orientation'
    threshold=float(platt.predict_proba(logit(np.array([t.threshold.iloc[0]])))[0,1])
    rawscore=t.score.to_numpy()
    t['raw_score']=rawscore;t['score']=platt.predict_proba(logit(rawscore))[:,1];t['threshold']=threshold
    t.to_parquet(out/'platt_test_predictions.parquet',index=False)
    v['raw_score']=v.score;v['score']=platt.predict_proba(logit(v.raw_score.to_numpy()))[:,1];v['threshold']=threshold
    v.to_parquet(out/'platt_validation_predictions.parquet',index=False)
    joblib.dump(platt,out/'platt_model.joblib')
    calibrated=evaluate(t.label.to_numpy(),t.score.to_numpy(),threshold)
    pd.DataFrame([{'model':'validation-fitted Platt','seed':42,**calibrated}]).to_csv(out/'calibration_metrics.csv',index=False)
    print('Primary controls complete',flush=True)


def horizons(data_dir,root):
    out=root/'horizons';out.mkdir(exist_ok=True)
    _,labels,inputs,outputs=reconstruction.load_raw(data_dir)
    rows=[];counts=[]
    for horizon in [1,2,4,8]:
        reconstruction.HORIZON=horizon
        wallets,events=reconstruction.build_wallet_index(labels,inputs,outputs)
        wallets=wallets[wallets.first_step<=42].copy()
        activity=reconstruction.activity_features(wallets,events)
        sna,_=reconstruction.pair_features(wallets,inputs,outputs)
        samples=activity.merge(sna,on='address').fillna(0)
        run=out/f'H{horizon}';run.mkdir(exist_ok=True)
        samples.to_parquet(run/'wallet_samples.parquet',index=False)
        train=samples.split=='train';val=samples.split=='validation';test=samples.split=='test'
        counts.append({'horizon':horizon,'train':int(train.sum()),'validation':int(val.sum()),'test':int(test.sum()),'illicit':int(samples.loc[test,'label'].sum())})
        for name,columns in [('activity',ACTIVITY),('activity_sna',ACTIVITY+SNA)]:
            model=xgb(samples.loc[train,'label'].to_numpy())
            model.fit(samples.loc[train,columns],samples.loc[train,'label'])
            vs=model.predict_proba(samples.loc[val,columns])[:,1]
            ts=model.predict_proba(samples.loc[test,columns])[:,1]
            threshold=select_f1_threshold(samples.loc[val,'label'].to_numpy(),vs)
            rows.append({'horizon':horizon,'features':name,'seed':42,**evaluate(samples.loc[test,'label'].to_numpy(),ts,threshold)})
            for split,mask,score in [('validation',val,vs),('test',test,ts)]:
                frame=samples.loc[mask,['address','label','first_step','cutoff']].copy()
                frame['score'],frame['threshold']=score,threshold
                frame.to_parquet(run/f'{name}_{split}_predictions.parquet',index=False)
            joblib.dump(model,run/f'{name}_model.joblib')
        print(f'Horizon{horizon} complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'cohort_counts.csv',index=False)


def transactions(data_dir,repo,root):
    out=root/'transactions';out.mkdir(exist_ok=True)
    features=pd.read_csv(data_dir/'txs_features.csv')
    labels=pd.read_csv(repo/'Transactions Dataset'/'txs_classes.csv')
    labels['class']=pd.to_numeric(labels['class'],errors='coerce')
    labels=labels[labels['class'].isin([1,2])]
    labels['label']=(labels['class']==1).astype(np.int8)
    edges=pd.read_csv(repo/'Transactions Dataset'/'txs_edgelist.csv').drop_duplicates()
    src_col,dst_col=edges.columns[:2]
    times=features.set_index('txId')['Time step']
    edges['source_time']=edges[src_col].map(times);edges['target_time']=edges[dst_col].map(times)
    incoming=edges[edges.source_time<=edges.target_time].copy()
    outgoing=edges[edges.target_time<=edges.source_time].copy()
    topology=features[['txId','Time step']].set_index('txId')
    topology['causal_in']=incoming.groupby(dst_col).size()
    topology['causal_out']=outgoing.groupby(src_col).size()
    topology=topology.fillna(0)
    topology['causal_total']=topology.causal_in+topology.causal_out
    topology['log_causal_in']=np.log1p(topology.causal_in)
    topology['log_causal_out']=np.log1p(topology.causal_out)
    incoming['age']=incoming.target_time-incoming.source_time
    topology['mean_predecessor_age']=incoming.groupby(dst_col).age.mean()
    topology['max_predecessor_age']=incoming.groupby(dst_col).age.max()
    incoming['predecessor_activity']=incoming[src_col].map(topology.causal_total)
    topology['mean_predecessor_activity']=incoming.groupby(dst_col).predecessor_activity.mean()
    topology=topology.fillna(0).drop(columns='Time step').reset_index()
    sna=[c for c in topology if c!='txId']
    features=features.merge(topology,on='txId',validate='one_to_one')
    # Supplied named out-degree can encode a later spend; replace both degree
    # fields, not only the added topology block.
    features['in_txs_degree']=features.causal_in
    features['out_txs_degree']=features.causal_out
    samples=features.merge(labels[['txId','label']],on='txId',validate='one_to_one')
    samples['split']=np.select([samples['Time step']<=29,samples['Time step']<=36],['train','validation'],default='test')
    samples.to_parquet(out/'transaction_samples.parquet',index=False)
    local=[f'Local_feature_{i}' for i in range(1,94)]
    named=['in_txs_degree','out_txs_degree','total_BTC','fees','size','num_input_addresses','num_output_addresses',
           'in_BTC_min','in_BTC_max','in_BTC_mean','in_BTC_median','in_BTC_total','out_BTC_min','out_BTC_max','out_BTC_mean','out_BTC_median','out_BTC_total']
    train=samples.split=='train';val=samples.split=='validation';test=samples.split=='test'
    rows=[];preds=[];vpreds=[]
    for view,base in [('Elliptic',local),('Elliptic++ tx',local+named)]:
      for seed in [13,42,97]:
       for name,columns in [('local',base),('local_causal_sna',base+sna)]:
        started=time.perf_counter();model=xgb(samples.loc[train,'label'].to_numpy(),seed)
        model.fit(samples.loc[train,columns],samples.loc[train,'label'])
        vs=model.predict_proba(samples.loc[val,columns])[:,1];ts=model.predict_proba(samples.loc[test,columns])[:,1]
        threshold=select_f1_threshold(samples.loc[val,'label'].to_numpy(),vs)
        rows.append({'view':view,'features':name,'seed':seed,**evaluate(samples.loc[test,'label'].to_numpy(),ts,threshold),
                     'run_timestamp':utc_now(),'elapsed_seconds':time.perf_counter()-started,'configuration_hash':config_hash({'view':view,'features':columns,'seed':seed})})
        for mask,score,collection in [(val,vs,vpreds),(test,ts,preds)]:
            frame=samples.loc[mask,['txId','label','Time step']].copy()
            frame['score'],frame['threshold'],frame['view'],frame['features'],frame['seed']=score,threshold,view,name,seed
            collection.append(frame)
        joblib.dump(model,out/f'{view.replace(" ","_")}_{name}_seed{seed}.joblib')
        print(f'Transaction {view} {name} seed{seed} done',flush=True)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.concat(preds).to_parquet(out/'test_predictions.parquet',index=False)
    pd.concat(vpreds).to_parquet(out/'validation_predictions.parquet',index=False)
    samples.groupby('split').label.agg(['size','sum','mean']).to_csv(out/'cohort_counts.csv')
    save_json(out/'manifest.json',{'raw_sha256':{str(f.name):sha256(f) for f in [data_dir/'txs_features.csv',repo/'Transactions Dataset'/'txs_classes.csv',repo/'Transactions Dataset'/'txs_edgelist.csv']},
       'local_columns':local,'named_columns':named,'sna_columns':sna,'causality':'incoming source_time<=target; outgoing target_time<=source; supplied named degrees replaced',
       'historical_note':'New audited reconstruction; original control predictions were not provided and the earlier rounded values are not reused.'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('analysis/locked_20261002'))
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--repo-dir',type=Path,required=True)
    args=p.parse_args();started=time.perf_counter()
    primary_controls(args.root);horizons(args.data_dir,args.root);transactions(args.data_dir,args.repo_dir,args.root)
    save_json(args.root/'controls'/'manifest.json',{'environment':environment(),'run_timestamp':utc_now(),
       'elapsed_seconds':time.perf_counter()-started,'script_sha256':sha256(Path(__file__)),
       'permutation_repeats':10,'label_shuffle_seeds':[1,2,3,4,5],'horizon_test_first_steps':[37,42],
       'calibration':'validation-fitted logistic over logits; original threshold monotonically mapped, not retuned on test'})

if __name__=='__main__':main()
