"""Independent metric reconstruction, publication tables and vector figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reconstruct_wallet_experiments import evaluate
from audit_utils import sha256, save_json, environment

METRICS=['ap','precision','recall','f1','roc_auc','brier','p_at_1pct']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('analysis/locked_20261002'))
    p.add_argument('--bundle',type=Path,default=Path('revision_bundle'));p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();root,bundle=args.root,args.bundle
    generated=bundle/'generated';figures=bundle/'figures'
    generated.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    rows=[];validation_checks=[]
    def add(frame,metadata,counts,expected=None,prefix=False):
        threshold=float(frame.threshold.iloc[0])
        result=evaluate(frame.label.to_numpy(),frame.score.to_numpy(),threshold)
        if expected is not None:
            for key in METRICS+['threshold']:
                assert np.isclose(result[key],float(expected[key]),rtol=2e-6,atol=1e-7),(metadata,key,result[key],expected[key])
            validation_checks.append({'dataset':metadata['dataset'],'model':metadata['model'],'features':metadata['feature_set'],'seed':metadata['seed'],'metrics_recomputed':True})
        rows.append({**metadata,'train_count':counts[0],'validation_count':counts[1],'test_count':len(frame),
                     'test_illicit':int(frame.label.sum()),'validation_threshold':threshold,**result})
    counts_csv=pd.read_csv(root/'primary'/'cohort_counts.csv').set_index('split')
    counts=[int(counts_csv.loc[s,'size']) for s in ['train','validation','test']]
    manifest=json.loads((root/'primary'/'manifest.json').read_text())
    primary=pd.read_csv(root/'primary'/'model_metrics.csv')
    predictions=pd.read_parquet(root/'primary'/'test_predictions.parquet')
    for _,metric in primary.iterrows():
        frame=predictions[(predictions.learner==metric.learner)&(predictions.seed==metric.seed)&(predictions.features==metric.features)]
        add(frame,{'dataset':'Elliptic++','task':'wallet H4','model':metric.learner,'feature_set':metric.features,'seed':metric.seed,
                   'run_timestamp':manifest['run_timestamp'],'configuration_hash':manifest['configuration_hash']},counts,metric)
    neural_frames=[]
    mlp_dir=root/'mlp_sensitivity'
    if (mlp_dir/'manifest.json').exists():
        mlp_metrics=pd.read_csv(mlp_dir/'metrics.csv');mlp_preds=pd.read_parquet(mlp_dir/'test_predictions.parquet')
        for _,metric in mlp_metrics.iterrows():
            frame=mlp_preds[(mlp_preds.seed==metric.seed)&(mlp_preds.features==metric.features)]
            add(frame,{'dataset':'Elliptic++','task':'wallet H4','model':'MLP','feature_set':metric.features,'seed':metric.seed,
                'run_timestamp':metric.run_timestamp,'configuration_hash':metric.configuration_hash,'elapsed_seconds':metric.elapsed_seconds},counts,metric)
    elif not args.allow_partial:
        raise RuntimeError('Incomplete tabular MLP sensitivity')
    for directory in sorted((root/'neural').glob('*')):
        if not (directory/'manifest.json').exists():
            if args.allow_partial:continue
            raise RuntimeError(f'Incomplete neural run: {directory}')
        metric=pd.read_csv(directory/'metrics.csv').iloc[0]
        m=json.loads((directory/'manifest.json').read_text())
        frame=pd.read_parquet(directory/'test_predictions.parquet')
        metadata={'dataset':'Elliptic++','task':'wallet H4','model':metric.model,'feature_set':metric.variant,
            'seed':metric.seed,'max_tokens':metric.max_tokens,'run_timestamp':metric.run_timestamp,'configuration_hash':m['configuration_hash'],
            'elapsed_seconds':metric.elapsed_seconds,'parameter_count':metric.parameter_count}
        add(frame,metadata,counts,metric)
        prefix=frame.copy();prefix['score']=prefix.prefix_score;prefix['threshold']=prefix.prefix_threshold
        add(prefix,{**metadata,'task':'wallet H2 prefix'},counts,{k:metric['prefix_'+k] for k in METRICS+['threshold']})
        neural_frames.append(metric)
    neural=pd.DataFrame(neural_frames)
    if not args.allow_partial:
        assert len(neural) == 21, f'Expected 21 completed TET runs, found {len(neural)}'
        for cap in [8,16,32]:
            selected=neural[(neural.variant=='pc_tet_activity_sna')&(neural.max_tokens==cap)]
            assert set(selected.seed)=={13,42,97}, f'Missing cap {cap} seeds'
        for family in ['ego','tgat']:
            complete=[d for d in (root/family).glob('seed*') if (d/'manifest.json').exists()]
            assert len(complete)==3, f'Expected three completed {family} runs'
    neural.to_csv(root/'neural_run_metrics.csv',index=False)
    for directory in sorted((root/'ego').glob('seed*')):
        if not (directory/'manifest.json').exists():continue
        metric_table=pd.read_csv(directory/'ego_model_metrics.csv')
        frames=pd.read_parquet(directory/'ego_test_predictions.parquet')
        m=json.loads((directory/'manifest.json').read_text())
        for _,metric in metric_table.iterrows():
            frame=frames[frames.variant==metric.variant]
            add(frame,{'dataset':'Elliptic++','task':'wallet H4','model':'CausalEgoSAGE','feature_set':metric.variant,'seed':metric.seed,
                'run_timestamp':metric.run_timestamp,'configuration_hash':m['configuration_hash'],
                'elapsed_seconds':metric.elapsed_seconds,'parameter_count':metric.parameter_count},counts,metric)
    for directory in sorted((root/'tgat').glob('seed*')):
        if not (directory/'manifest.json').exists():continue
        metric=pd.read_csv(directory/'metrics.csv').iloc[0]
        add(pd.read_parquet(directory/'test_predictions.parquet'),{'dataset':'Elliptic++','task':'wallet H4','model':'TGAT (causal adaptation)',
            'feature_set':'activity_sna + two-hop events','seed':metric.seed,'run_timestamp':metric.run_timestamp,
            'configuration_hash':metric.configuration_hash,'elapsed_seconds':metric.elapsed_seconds,'parameter_count':metric.parameter_count},counts,metric)
    ext=pd.read_csv(root/'external'/'real_cats_external_metrics.csv')
    extpred=pd.read_parquet(root/'external'/'test_predictions.parquet')
    extmanifest=json.loads((root/'external'/'manifest.json').read_text())
    extcounts=[extmanifest['train_count'],extmanifest['validation_count'],extmanifest['external_rows']]
    for _,metric in ext.iterrows():
        frame=extpred[(extpred.learner==metric.learner)&(extpred.seed==metric.seed)&(extpred.features==metric.features)]
        add(frame,{'dataset':'Real-CATS to Sup-CATS','task':'completed-profile transfer','model':metric.learner,'feature_set':metric.features,'seed':metric.seed,
            'run_timestamp':metric.run_timestamp,'configuration_hash':metric.configuration_hash,'elapsed_seconds':metric.elapsed_seconds},extcounts,metric)
    tx=pd.read_csv(root/'transactions'/'metrics.csv')
    txpred=pd.read_parquet(root/'transactions'/'test_predictions.parquet')
    txcounts=pd.read_csv(root/'transactions'/'cohort_counts.csv').set_index('split')
    for _,metric in tx.iterrows():
        frame=txpred[(txpred['view']==metric['view'])&(txpred.seed==metric.seed)&(txpred.features==metric.features)]
        add(frame,{'dataset':metric['view'],'task':'transaction creation','model':'XGBoost','feature_set':metric.features,'seed':metric.seed,
            'run_timestamp':metric.run_timestamp,'configuration_hash':metric.configuration_hash},[int(txcounts.loc[s,'size']) for s in ['train','validation','test']],metric)
    for _,metric in pd.read_csv(root/'horizons'/'metrics.csv').iterrows():
        directory=root/'horizons'/f'H{int(metric.horizon)}'
        c=pd.read_csv(root/'horizons'/'cohort_counts.csv').set_index('horizon').loc[metric.horizon]
        add(pd.read_parquet(directory/f'{metric.features}_test_predictions.parquet'),{'dataset':'Elliptic++','task':f'paired-wallet H{int(metric.horizon)}',
            'model':'XGBoost','feature_set':metric.features,'seed':42,'run_timestamp':manifest['run_timestamp'],
            'configuration_hash':sha256(directory/'wallet_samples.parquet')},[int(c['train']),int(c.validation),int(c.test)],metric)
    calibration=pd.read_csv(root/'controls'/'calibration_metrics.csv').iloc[0]
    add(pd.read_parquet(root/'controls'/'platt_test_predictions.parquet'),{'dataset':'Elliptic++','task':'wallet H4','model':'XGBoost + validation Platt',
        'feature_set':'activity_sna','seed':42,'run_timestamp':manifest['run_timestamp'],'configuration_hash':sha256(root/'controls'/'platt_model.joblib')},counts,calibration)
    permutation=pd.read_csv(root/'controls'/'label_permutation_metrics.csv')
    perm_pred=pd.read_parquet(root/'controls'/'label_permutation_test_predictions.parquet')
    for _,metric in permutation.iterrows():
        add(perm_pred[perm_pred.seed==metric.seed],{'dataset':'Elliptic++','task':'wallet H4 label shuffle','model':'XGBoost','feature_set':'activity_sna',
            'seed':metric.seed,'run_timestamp':manifest['run_timestamp'],'configuration_hash':sha256(root/'controls'/'manifest.json')},counts,metric)
    recency=pd.read_csv(root/'primary'/'recency_weighting.csv')
    rp=pd.read_parquet(root/'primary'/'recency_test_predictions.parquet')
    for _,metric in recency.iterrows():
        add(rp[rp.decay==metric.decay],{'dataset':'Elliptic++','task':'wallet H4 recency','model':'XGBoost','feature_set':'activity_sna','seed':42,
            'recency_lambda':metric.decay,'run_timestamp':manifest['run_timestamp'],'configuration_hash':manifest['configuration_hash']},counts,metric)
    master=pd.DataFrame(rows);master.to_csv(root/'MASTER_RESULTS.csv',index=False)
    save_json(root/'METRIC_VERIFICATION.json',{'verified_rows':len(validation_checks),'checks':validation_checks,'environment':environment()})
    summary=master.groupby(['dataset','task','model','feature_set','max_tokens'],dropna=False)[METRICS].agg(['mean','std'])
    summary.to_csv(root/'SUMMARY_RESULTS.csv')
    print(f'Verified {len(validation_checks)} metric rows from saved predictions',flush=True)
    # Verify token truncation by class and split, with pair-event weighted retention.
    samples=pd.read_parquet(root/'primary'/'wallet_samples.parquet')
    pairs=pd.read_parquet(root/'primary'/'neural_views_v2'/'full_pairs.parquet')
    pairs=pairs.sort_values(['address','pair_total','relative_first','counterparty'],ascending=[True,False,True,True],kind='stable')
    pairs['rank']=pairs.groupby('address').cumcount()
    total=pairs.groupby('address').agg(counterparties=('pair_total','size'),events=('pair_total','sum'))
    token_rows=[]
    for cap in [8,16,32]:
        retained=pairs[pairs['rank']<cap].groupby('address').pair_total.sum()
        data=samples[['address','label','split']].merge(total,on='address',how='left').fillna(0)
        data['retained']=data.address.map(retained).fillna(0)
        for (split,label),group in data.groupby(['split','label']):
            token_rows.append({'cap':cap,'split':split,'label':label,'wallets':len(group),'over_cap_fraction':float((group.counterparties>cap).mean()),
                               'pair_event_retention':float(group.retained.sum()/group.events.sum()),
                               'pair_event_count':float(group.events.sum()),'retained_event_count':float(group.retained.sum())})
    token_audit=pd.DataFrame(token_rows);token_audit.to_csv(root/'TOKEN_CAP_AUDIT.csv',index=False)
    # Publication tables, generated from verified rows, never typed cached means.
    def cell(group,key):
        values=group[key].astype(float)
        return f'${values.mean():.4f}\\!\\pm\\!{values.std(ddof=1):.4f}$' if len(values)>1 else f'{values.iloc[0]:.4f}'
    def table_rows(path,records,keys,stacked=False):
        lines=[]
        for label,frame in records:
            values=[(r'\shortstack{'+f'{frame[k].mean():.4f}'+r'\\$\pm$'+f'{frame[k].std(ddof=1):.4f}'+'}' if stacked and len(frame)>1 else cell(frame,k)) for k in keys]
            lines.append(' & '.join([label]+values))
        separator=(r' \\[4pt]' if stacked else r' \\')+'\n'
        path.write_text(separator.join(lines)+'\n',encoding='utf-8')
    primary_lines=[]
    for learner in ['XGBoost','HGB']:
        for feature in ['activity','activity_sna']:
            frame=primary[(primary.learner==learner)&(primary.features==feature)]
            if len(frame):
                values=[cell(frame,k) for k in ['ap','f1']]+[f'{frame[k].mean():.4f}' for k in ['precision','recall','roc_auc','brier','p_at_1pct']]
                primary_lines.append(' & '.join([learner,'Activity' if feature=='activity' else 'Activity + causal SNA']+values)+r' \\')
    primary_lines[-1]=primary_lines[-1][:-3]
    (generated/'primary_rows.tex').write_text('\n'.join(primary_lines)+'\n',encoding='utf-8')
    if len(neural):
        normal=neural[neural.max_tokens==16]
        records=[]
        for variant,label in [('tet_activity_sna','TET-4'),('dual_tet_activity_sna','Dual-horizon TET'),('cons_tet_activity_sna','Consistency TET'),('pc_tet_activity_sna','PC-TET')]:
            frame=normal[normal.variant==variant]
            if len(frame):records.append((label,frame))
        table_rows(generated/'objective_ablation_rows.tex',records,['ap','f1','p_at_1pct','brier','prefix_ap','prefix_f1','prefix_p_at_1pct','prefix_brier','mean_full_prefix_gap'],stacked=True)
        table_rows(generated/'score_diagnostics_rows.tex',records,['score_sd','prefix_score_sd','ece_10bins','prefix_ece_10bins'])
        cap_records=[]
        for cap in [8,16,32]:
            frame=neural[(neural.variant=='pc_tet_activity_sna')&(neural.max_tokens==cap)]
            if len(frame):cap_records.append((str(cap),frame))
        table_rows(generated/'token_sensitivity_rows.tex',cap_records,['ap','prefix_ap','brier','mean_full_prefix_gap'])
    table_rows(generated/'external_rows.tex',[(f'{learner} & '+('Activity' if feature=='activity' else 'Activity + relation'),ext[(ext.learner==learner)&(ext.features==feature)]) for learner in ['XGBoost','MLP'] for feature in ['activity','activity_relation']],['ap','f1','roc_auc','brier','p_at_1pct'])
    table_rows(generated/'transaction_rows.tex',[(f'{view} & '+('Local' if feature=='local' else '+ causal SNA'),tx[(tx.view==view)&(tx.features==feature)]) for view in ['Elliptic','Elliptic++ tx'] for feature in ['local','local_causal_sna']],['ap','f1','roc_auc'])
    neural_records=[]
    wallet_master=master[(master.dataset=='Elliptic++')&(master.task=='wallet H4')]
    for model,feature,label in [('MLP','activity','MLP & Activity'),('MLP','activity_sna','MLP & Activity + SNA'),
        ('CausalEgoSAGE','ego_activity','CausalEgoSAGE & Activity + local ego'),('CausalEgoSAGE','ego_activity_sna','CausalEgoSAGE & Activity + local ego + SNA'),
        ('TET','tet_activity','TET & Activity + temporal tokens'),('TET','tet_activity_sna','TET & Activity + temporal tokens + SNA'),
        ('PC-TET','pc_tet_activity_sna','PC-TET & Activity + temporal tokens + SNA'),('TGAT (causal adaptation)','activity_sna + two-hop events','TGAT & Activity + SNA + two-hop events')]:
        frame=wallet_master[(wallet_master.model==model)&(wallet_master.feature_set==feature)&((wallet_master.max_tokens==16)|wallet_master.max_tokens.isna())]
        if len(frame):neural_records.append((label,frame))
    table_rows(generated/'neural_rows.tex',neural_records,['ap','f1','roc_auc','brier','p_at_1pct'])
    # Charts share typography and do not conceal negative results.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    colors=['#8299b5','#164e79']
    fig,axes=plt.subplots(1,2,figsize=(9,3.1),layout='constrained')
    for ax,view,data in [(axes[0],'Wallet addresses',primary[primary.learner=='XGBoost']),(axes[1],'Linked transaction controls',tx)]:
        if ax is axes[0]:
            for j,feature in enumerate(['activity','activity_sna']):
                d=data[data.features==feature][['ap','f1','p_at_1pct']]
                ax.bar(np.arange(3)+(j-.5)*.33,d.mean(),.33,yerr=d.std(ddof=1),capsize=3,color=colors[j],label=['Activity','Activity + causal SNA'][j])
            ax.set_xticks(range(3),['AP','F1','P@1%']);ax.set_ylim(0,.7)
        else:
            for j,feature in enumerate(['local','local_causal_sna']):
                values=[data[(data.view==v)&(data.features==feature)].ap for v in ['Elliptic','Elliptic++ tx']]
                ax.bar(np.arange(2)+(j-.5)*.33,[v.mean() for v in values],.33,yerr=[v.std(ddof=1) for v in values],capsize=3,color=colors[j],label=['Local','Local + causal SNA'][j])
            ax.set_xticks(range(2),['Elliptic','Elliptic++ tx']);ax.set_ylabel('Average precision');ax.set_ylim(0,.85)
        ax.set_title(view);ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(figures/'fig3_main_results.pdf');plt.close(fig)
    h=pd.read_csv(root/'horizons'/'metrics.csv')
    fig,ax=plt.subplots(figsize=(4.7,2.8),layout='constrained')
    for j,feature in enumerate(['activity','activity_sna']):
        d=h[h.features==feature];ax.plot(d.horizon,d.ap,'o-',label=['Activity','Activity + causal SNA'][j],color=colors[j])
    ax.set(xlabel='Observation horizon (time steps)',ylabel='Average precision',xticks=[1,2,4,8],ylim=(0,.45));ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.savefig(figures/'fig4_horizon_sensitivity.pdf');plt.close(fig)
    imp=pd.read_csv(root/'controls'/'permutation_importance.csv').groupby('feature').ap_decrease.agg(['mean','std']).sort_values('mean').tail(8)
    fig,ax=plt.subplots(figsize=(4.7,3),layout='constrained')
    ax.barh(imp.index.str.replace('_',' '),imp['mean'],xerr=imp['std'],color=colors[1],capsize=2)
    ax.set(xlabel='Validation AP decrease',title='Fixed-horizon wallet features');ax.grid(axis='x',alpha=.2)
    fig.savefig(figures/'fig5_feature_importance.pdf');plt.close(fig)
    # Explanatory diagrams are vector-authored from the stated protocol.
    def box(ax,x,y,w,h,text,color='#edf3f8'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor=color,edgecolor='#33536e',linewidth=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    fig,ax=plt.subplots(figsize=(10,3.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.02,.91,'Potential non-causal path',weight='bold',fontsize=11)
    ax.text(.02,.44,'Proposed leakage-safe path',weight='bold',fontsize=11)
    top=['Completed graph and\nlifetime summaries','Duplicate/repeated\nwallet rows','Row-based partition\nand test-led selection','Retrospective\nrecognition']
    bottom=['Deduplicated\naddress events','First appearance\none cohort per wallet','Wallet-specific cutoff\ncausal features','Train then validation\nfreeze and future score']
    for y,items in [(.58,top),(.1,bottom)]:
        for i,text in enumerate(items):
            x=.025+i*.25;box(ax,x,y,.205,.24,text,'#faeded' if y>.5 else '#edf3f8')
            if i<3:ax.annotate('',xy=(x+.242,y+.12),xytext=(x+.21,y+.12),arrowprops={'arrowstyle':'->','color':'#33536e'})
    fig.savefig(figures/'fig1_protocol_audit.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,2.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.scatter([.3,.08,.12,.5,.62],[.5,.8,.15,.8,.2],s=[1000,500,500,500,500],c=['#164e79']+['#a9c3d9']*4)
    for x,y in [(.08,.8),(.12,.15),(.5,.8)]:ax.annotate('',xy=(x,y),xytext=(.3,.5),arrowprops={'arrowstyle':'->','color':'#164e79','lw':2})
    ax.annotate('',xy=(.62,.2),xytext=(.3,.5),arrowprops={'arrowstyle':'->','color':'#888','linestyle':'--','lw':2})
    ax.text(.3,.5,'Wallet',ha='center',va='center',color='white',fontsize=9)
    ax.text(.72,.62,'Visible interactions\ntime <= wallet cutoff',fontsize=9)
    ax.text(.72,.24,'Later interaction\nexcluded',fontsize=9,color='#666')
    fig.savefig(figures/'fig2_causal_actor_snapshot.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.3));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for y,horizon in [(.68,2),(.13,4)]:
        box(ax,.01,y,.22,.21,f'{horizon}-step causal view\nindependently reconstructed')
        box(ax,.29,y,.2,.21,'Activity and SNA\ncounterparty tokens')
        ax.annotate('',xy=(.285,y+.105),xytext=(.235,y+.105),arrowprops={'arrowstyle':'->'})
        ax.annotate('',xy=(.55,.5),xytext=(.5,y+.105),arrowprops={'arrowstyle':'->'})
    box(ax,.55,.36,.18,.27,'Shared TET encoder\nmasked attention')
    box(ax,.80,.65,.17,.2,'Prefix probability p2')
    box(ax,.80,.15,.17,.2,'Full probability p4')
    for y in [.75,.25]:ax.annotate('',xy=(.795,y),xytext=(.735,.5),arrowprops={'arrowstyle':'->'})
    ax.text(.85,.47,'BCE4 + 0.5 BCE2\n+ 0.2 (p4 - p2)^2',ha='center',fontsize=9)
    ax.text(.55,.01,'Checkpoint: full-view validation AP. Thresholds: matching validation views only.',ha='center',fontsize=9)
    fig.savefig(figures/'fig6_pctet_architecture.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.5,.93,'Early screening of wallet addresses',ha='center',weight='bold',fontsize=15)
    box(ax,.02,.26,.24,.5,'First appearance\n4-step causal history\none sample per address')
    box(ax,.34,.26,.3,.5,'Activity + counterparty network\nAP 0.1360 to 0.3508\nF1 0.2587 to 0.4370\nP@1% 0.2157 to 0.5882')
    box(ax,.71,.26,.27,.5,'Shared two-view TET\nobjective-component ablation\nTemporal shift persists\nExternal top-alert failure')
    for x in [.26,.64]:ax.annotate('',xy=(x+.073,.51),xytext=(x+.006,.51),arrowprops={'arrowstyle':'->','lw':1.7})
    ax.text(.5,.06,'Chronological holdout previously examined; completed-profile transfer is not a causal replication.',ha='center',fontsize=9)
    fig.savefig(figures/'graphical_abstract.pdf',bbox_inches='tight');fig.savefig(figures/'graphical_abstract.png',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
