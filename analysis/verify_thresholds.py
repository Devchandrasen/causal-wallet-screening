"""Independently reconstruct validation thresholds and check disjoint scoring identities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from audit_utils import save_json, utc_now

def independent_threshold(y,score,exact=False):
    # Ascending group sums give TP and FP for score >= each candidate.
    y=np.asarray(y,dtype=np.int64)
    grouped=pd.DataFrame({'score':score,'y':y}).groupby('score',sort=True).agg(tp=('y','sum'),n=('y','size'))
    candidates=grouped.index.to_numpy()
    if not exact and len(candidates)>4000:candidates=np.quantile(score,np.linspace(0,1,4001))
    cumulative=grouped.iloc[::-1].cumsum().iloc[::-1]
    indices=np.searchsorted(grouped.index.to_numpy(),candidates,side='left')
    tp=cumulative.tp.to_numpy()[indices];n=cumulative.n.to_numpy()[indices]
    precision=tp/n;recall=tp/max(1,int(y.sum()))
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(len(tp),dtype=float),where=precision+recall>0)
    return float(candidates[np.argmax(f1)])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();root=args.root;checks=[]
    def check(vpath,tpath,keys=(),exact=False,prefix=False,test_filter=None):
        v=pd.read_parquet(vpath);t=pd.read_parquet(tpath)
        if test_filter:
            for key,value in test_filter.items():t=t[t[key]==value]
        groups=[(None,v)] if not keys else v.groupby(list(keys),dropna=False)
        for identity,frame in groups:
            other=t
            if keys:
                values=identity if isinstance(identity,tuple) else (identity,)
                for key,value in zip(keys,values):other=other[other[key]==value]
            idcol='address' if 'address' in frame else 'txId'
            assert frame[idcol].is_unique and other[idcol].is_unique
            assert not (set(frame[idcol])&set(other[idcol]))
            for name in ['score']+(['prefix_score'] if prefix else []):
                column='threshold' if name=='score' else 'prefix_threshold'
                th=independent_threshold(frame.label.to_numpy(),frame[name].to_numpy(),exact)
                assert np.isclose(th,frame[column].iloc[0],rtol=0,atol=1e-12),(vpath,identity,name,th,frame[column].iloc[0])
                assert np.isclose(th,other[column].iloc[0],rtol=0,atol=1e-12)
                checks.append({'validation_file':str(vpath.relative_to(root)),'group':str(identity),'view':name,'threshold':th,'threshold_recomputed':True,'identities_disjoint':True})
    check(root/'primary/validation_predictions.parquet',root/'primary/test_predictions.parquet',['learner','seed','features'])
    check(root/'primary/recency_validation_predictions.parquet',root/'primary/recency_test_predictions.parquet',['decay'])
    check(root/'mlp_sensitivity/validation_predictions.parquet',root/'mlp_sensitivity/test_predictions.parquet',['seed','features'])
    for d in (root/'neural').iterdir():
        if d.is_dir():check(d/'validation_predictions.parquet',d/'test_predictions.parquet',prefix=True)
    for d in (root/'tgat').glob('seed*'):check(d/'validation_predictions.parquet',d/'test_predictions.parquet')
    for d in (root/'ego').glob('seed*'):
        for name in ['ego_activity','ego_activity_sna']:
            check(d/f'{name}_validation_predictions.parquet',d/'ego_test_predictions.parquet',test_filter={'variant':name})
    check(root/'external/validation_predictions.parquet',root/'external/test_predictions.parquet',['learner','seed','features'],exact=True)
    check(root/'transactions/validation_predictions.parquet',root/'transactions/test_predictions.parquet',['view','seed','features'])
    for horizon in [1,2,4,8]:
        d=root/'horizons'/f'H{horizon}'
        for name in ['activity','activity_sna']:check(d/f'{name}_validation_predictions.parquet',d/f'{name}_test_predictions.parquet')
    check(root/'controls/label_permutation_validation_predictions.parquet',root/'controls/label_permutation_test_predictions.parquet',['seed'])
    # Platt's mapped threshold is intentionally not a new F1 search.
    save_json(root/'THRESHOLD_VERIFICATION.json',{'timestamp':utc_now(),'verified_thresholds':len(checks),'checks':checks,
        'calibration_exception':'Platt threshold monotonically maps the source validation threshold; it is not retuned.'})
    print(f'Independently verified {len(checks)} validation thresholds and validation/test identity separation')

if __name__=='__main__':main()
