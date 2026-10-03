"""Portable sequential source-to-publication rerun, without external writes."""
import argparse
import subprocess
import sys
from pathlib import Path
from audit_utils import config_hash, environment, save_json, sha256, utc_now

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--elliptic-data',type=Path,required=True)
    p.add_argument('--elliptic-repo',type=Path,required=True)
    p.add_argument('--real-cats',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('analysis/locked_20261002'))
    p.add_argument('--bundle',type=Path,default=Path('revision_bundle'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();scripts=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=True)
    config=scripts/'locked_revision_config.json'
    save_json(args.output/'PIPELINE_INVOCATION.json',{'args':vars(args),'environment':environment(),
        'timestamp':utc_now(),'locked_config_sha256':sha256(config),
        'note':'This is an additional evaluation on an already examined chronological holdout, not a fresh confirmatory study.'})
    def run(script,*flags):
        subprocess.run([sys.executable,'-u',str(scripts/script),*map(str,flags)],check=True)
    if not args.resume or not (args.output/'primary'/'manifest.json').exists():
        run('reconstruct_wallet_experiments.py','--data-dir',args.elliptic_data,'--output-dir',args.output/'primary')
    run('run_neural_campaign.py','--root',args.output)
    run('run_cap_campaign.py','--root',args.output)
    run('run_mlp_sensitivity.py','--root',args.output)
    run('run_tgat_campaign.py','--root',args.output,'--data-dir',args.elliptic_data)
    if not args.resume or not (args.output/'external'/'manifest.json').exists():
        run('real_cats_external_validation.py','--data-dir',args.real_cats,'--output-dir',args.output/'external')
    if not args.resume or not (args.output/'controls'/'manifest.json').exists():
        run('rerun_controls.py','--root',args.output,'--data-dir',args.elliptic_data,'--repo-dir',args.elliptic_repo)
    run('bootstrap_revised_primary.py','--predictions',args.output/'primary'/'test_predictions.parquet','--output',args.output/'controls'/'paired_bootstrap.csv')
    run('summarize_locked_results.py','--root',args.output,'--bundle',args.bundle)
    run('verify_thresholds.py','--root',args.output)
    run('verify_provenance.py','--root',args.output,'--elliptic-data',args.elliptic_data,'--elliptic-repo',args.elliptic_repo,'--real-cats',args.real_cats)
    save_json(args.output/'PIPELINE_COMPLETE.json',{'timestamp':utc_now(),'master_sha256':sha256(args.output/'MASTER_RESULTS.csv'),
        'config_sha256':sha256(config),'script_sha256':{f.name:sha256(f) for f in scripts.glob('*.py')
        if f.name != 'generate_main_results_figure.py'}})

if __name__=='__main__':main()
