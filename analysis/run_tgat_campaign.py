import subprocess
import sys
import argparse
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('analysis/locked_20261002'))
parser.add_argument('--data-dir',type=Path,required=True)
args=parser.parse_args()
root=args.root
script=Path(__file__).resolve().parent/'train_causal_tgat.py'
for seed in (13,42,97):
    out=root/'tgat'/f'seed{seed}'
    if (out/'manifest.json').exists():continue
    out.mkdir(parents=True,exist_ok=True)
    print(f'START TGAT seed{seed}',flush=True)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u',str(script),'--data-dir',str(args.data_dir),
            '--results-dir',str(root/'primary'),'--output-dir',str(out),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'TGAT seed{seed} failed: {out / "run.log"}')
    print(f'DONE TGAT seed{seed}',flush=True)
