"""Serial audits after the locked full-budget pair has finished all dev work."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('runs/fineweb_energy_20260922')
    marker=root/'confirmation_full_budget.json'
    while not marker.exists():time.sleep(5)
    confirmation=json.loads(marker.read_text())
    if any(row['raw_byte_exposures']!=10_000_000 for row in confirmation['arms'].values()):
        raise ValueError('matched full-budget endpoints required')
    jobs=[
        ('fatigue_budget_v2','scripts.probe_energy_budget',[
            '--parent',str(root/'equilibrium_pair/equilibrium/checkpoint.pt'),
            '--out',str(root/'equilibrium_pair/fatigue_budget_v2.json')]),
        ('equilibrium_levels','scripts.probe_equilibrium_levels',[
            '--parent',str(root/'equilibrium_pair/equilibrium/checkpoint.pt'),
            '--out',str(root/'equilibrium_pair/conditional_levels.json')]),
        ('ridge_full_consumers','scripts.audit_fineweb_adapters',[
            '--parent',str(root/'ridge_metric8/checkpoint.pt'),
            '--out',str(root/'ridge_metric8/consumer_audit_full_budget.json')]),
        ('independent_test','scripts.evaluate_fineweb_test',[]),
        ('binding_full_budget','scripts.probe_fineweb_binding',[
            '--parents',str(root/'base8/checkpoint.pt'),str(root/'ridge_metric8/checkpoint.pt'),
            '--out',str(root/'binding_full_budget.json')]),
    ]
    for name,module,args in jobs:
        print(json.dumps(dict(event='start',job=name)),flush=True)
        with (root/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,'-m',module,*args],stdout=log,stderr=subprocess.STDOUT,
                           check=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
        print(json.dumps(dict(event='finished',job=name)),flush=True)


if __name__=='__main__':main()
