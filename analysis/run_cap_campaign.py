"""Expanded descriptive cap check after seed42 revealed cap sensitivity."""
import subprocess
import sys
import argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('analysis/locked_20261002'))
root=parser.parse_args().root
script=Path(__file__).resolve().parent/'train_prefix_consistent_transformer.py'
for cap in (8,32):
 for seed in (13,97):
    out=root/'neural'/f'pc_tet_activity_sna_seed{seed}_cap{cap}'
    if (out/'manifest.json').exists():continue
    out.mkdir(parents=True,exist_ok=True)
    print(f'START cap{cap} seed{seed}',flush=True)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u',str(script),'--results-dir',str(root/'primary'),
            '--output-dir',str(out),'--seed',str(seed),'--variant','pc_tet_activity_sna','--max-tokens',str(cap)],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'cap{cap} seed{seed} failed')
    print(f'DONE cap{cap} seed{seed}',flush=True)
