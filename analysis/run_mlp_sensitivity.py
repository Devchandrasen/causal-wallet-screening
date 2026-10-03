"""Complete the matched tabular MLP comparison beyond the original seed42 control."""
import argparse
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from reconstruct_wallet_experiments import ACTIVITY, SNA, evaluate, select_f1_threshold
from audit_utils import save_json, environment, utc_now, config_hash, sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    out=args.root/'mlp_sensitivity';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():return
    samples=pd.read_parquet(args.root/'primary'/'wallet_samples.parquet')
    train=samples.split.eq('train');val=samples.split.eq('validation');test=samples.split.eq('test')
    y=samples.label.to_numpy();records=[];predictions=[];validations=[]
    cfg={'seeds':[13,97],'hidden_layer_sizes':[64,32],'alpha':1e-4,'batch_size':1024,
         'learning_rate_init':1e-3,'max_iter':100,'early_stopping':True,'validation_fraction':.1,'n_iter_no_change':10}
    for seed in [13,97]:
        for feature,cols in [('activity',ACTIVITY),('activity_sna',ACTIVITY+SNA)]:
            start=time.perf_counter();scaler=StandardScaler()
            x=scaler.fit_transform(samples.loc[train,cols]);xv=scaler.transform(samples.loc[val,cols]);xt=scaler.transform(samples.loc[test,cols])
            model=MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-4,batch_size=1024,learning_rate_init=1e-3,
                max_iter=100,early_stopping=True,validation_fraction=.1,n_iter_no_change=10,random_state=seed)
            model.fit(x,y[train],sample_weight=compute_sample_weight('balanced',y[train]))
            vs=model.predict_proba(xv)[:,1];ts=model.predict_proba(xt)[:,1];th=select_f1_threshold(y[val],vs)
            records.append({'learner':'MLP','seed':seed,'features':feature,'run_timestamp':utc_now(),
                'configuration_hash':config_hash(cfg),'elapsed_seconds':time.perf_counter()-start,**evaluate(y[test],ts,th)})
            for mask,scores,target in [(val,vs,validations),(test,ts,predictions)]:
                frame=samples.loc[mask,['address','label','first_step','cutoff']].copy()
                frame['score'],frame['threshold'],frame['learner'],frame['seed'],frame['features']=scores,th,'MLP',seed,feature
                target.append(frame)
            joblib.dump({'model':model,'scaler':scaler,'columns':cols,'threshold':th},out/f'MLP_{feature}_seed{seed}.joblib')
            print(f'DONE MLP {feature} seed{seed}',flush=True)
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_parquet(out/'test_predictions.parquet',index=False)
    pd.concat(validations).to_parquet(out/'validation_predictions.parquet',index=False)
    save_json(out/'manifest.json',{'configuration':cfg,'configuration_hash':config_hash(cfg),'environment':environment(),
        'run_timestamp':utc_now(),'script_sha256':sha256(Path(__file__)),'note':'Seed42 is saved in primary; early stopping uses an internal stratified training subset, thresholds use chronological validation.'})

if __name__=='__main__':main()
