"""Sequential GPU campaign. Resume only completed, manifested runs."""
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('analysis/locked_20261002'))
    args = p.parse_args()
    script_dir = Path(__file__).resolve().parent
    jobs = []
    for seed in (13, 42, 97):
        for variant in ('tet_activity_sna', 'dual_tet_activity_sna', 'cons_tet_activity_sna', 'pc_tet_activity_sna', 'tet_activity'):
            out = args.root / 'neural' / f'{variant}_seed{seed}'
            jobs.append((out, 'train_prefix_consistent_transformer.py', ['--seed', str(seed), '--variant', variant]))
    for cap in (8, 32):
        out = args.root / 'neural' / f'pc_tet_activity_sna_seed42_cap{cap}'
        jobs.append((out, 'train_prefix_consistent_transformer.py', ['--seed', '42', '--variant', 'pc_tet_activity_sna', '--max-tokens', str(cap)]))
    for seed in (13, 42, 97):
        jobs.append((args.root / 'ego' / f'seed{seed}', 'train_causal_ego_encoder.py', ['--seed', str(seed)]))
    for out, script, flags in jobs:
        if (out / 'manifest.json').exists():
            print(f'SKIP completed {out.name}', flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(script_dir/script), '--results-dir', str(args.root/'primary'), '--output-dir', str(out), *flags]
        print(f'START {out.name}', flush=True)
        with (out/'run.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{out.name} failed; inspect {out / "run.log"}')
        print(f'DONE {out.name}', flush=True)

if __name__ == '__main__':
    main()
