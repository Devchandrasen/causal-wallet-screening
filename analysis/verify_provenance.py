"""Check raw-file and fitted-run source hashes without modifying experiments."""
import argparse
import json
from pathlib import Path
from audit_utils import save_json, sha256, utc_now

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--elliptic-data',type=Path,required=True);p.add_argument('--elliptic-repo',type=Path,required=True)
    p.add_argument('--real-cats',type=Path,required=True);args=p.parse_args();scripts=Path(__file__).resolve().parent;checks=[]
    families={'primary':'reconstruct_wallet_experiments.py','neural':'train_prefix_consistent_transformer.py',
        'ego':'train_causal_ego_encoder.py','tgat':'train_causal_tgat.py','external':'real_cats_external_validation.py',
        'controls':'rerun_controls.py','mlp_sensitivity':'run_mlp_sensitivity.py'}
    for path in args.root.rglob('manifest.json'):
        parts=path.relative_to(args.root).parts
        if parts[0] not in families:continue
        manifest=json.loads(path.read_text())
        if 'script_sha256' not in manifest:continue # derived query cache, not a fitted run
        actual=sha256(scripts/families[parts[0]])
        assert actual==manifest['script_sha256'],path
        checks.append({'manifest':str(path.relative_to(args.root)),'source_hash_verified':True})
    assert len(checks)==31,len(checks)
    raw=[]
    for directory,source in [('primary',args.elliptic_data),('external',args.real_cats),('transactions',args.elliptic_data)]:
        manifest=json.loads((args.root/directory/'manifest.json').read_text())
        for filename,expected in manifest.get('raw_sha256',{}).items():
            location=source/filename
            if directory=='transactions' and filename in ['txs_classes.csv','txs_edgelist.csv']:
                location=args.elliptic_repo/'Transactions Dataset'/filename
            actual=sha256(location);assert actual==expected,location
            raw.append({'run_family':directory,'file':filename,'sha256':actual,'verified':True})
    save_json(args.root/'PROVENANCE_VERIFICATION.json',{'timestamp':utc_now(),'fitted_source_checks':checks,'raw_source_checks':raw})
    print(f'Verified {len(checks)} fitted-run source hashes and {len(raw)} raw-source hashes')

if __name__=='__main__':main()
