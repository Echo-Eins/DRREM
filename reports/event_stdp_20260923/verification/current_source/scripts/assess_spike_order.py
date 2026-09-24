"""Lock temporal-order models, then evaluate new nuisance sequences once."""
import argparse
import json
from pathlib import Path

import torch

from drrem.data.protocol import file_digest
from drrem.spiking_rrem import SpikingRREM
from scripts.probe_spike_order import make_batch,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/event_stdp_20260923'))
    a=p.parse_args();torch.set_num_threads(2)
    names=['order_frozen','order_resume','order_fixed_readout']
    plan={'cases':{n:file_digest(a.root/n/'checkpoint.pt') for n in names},
          'seeds':list(range(95000,95016)),'batch':32,'tail':4}
    dst=a.root/'order_final';dst.mkdir()
    (dst/'plan.json').write_text(json.dumps(plan,indent=2))
    # This is a distinct, diagnostic test set. No training/model choice follows.
    batches=[make_batch(seed,32,4) for seed in plan['seeds']]
    train_hashes=set()
    for seed in range(10000,10500):
        train_hashes.update(tuple(row) for row in make_batch(seed,32,4).x.tolist())
    assert not any(tuple(row) in train_hashes for b in batches for row in b.x.tolist())
    results={}
    for n in names:
        m=SpikingRREM.from_checkpoint(torch.load(a.root/n/'checkpoint.pt',map_location='cpu',weights_only=False),'cpu')
        results[n]=score(m,batches)
        results[n]['reset_history']=score(m,batches,True)
        if n=='order_fixed_readout':
            initial=SpikingRREM(m.cfg)
            assert torch.equal(initial.E,m.E) and torch.equal(initial.E_in,m.E_in)
            with torch.no_grad():initial.E_bias.fill_(-20.);initial.E_bias[:,48:50]=0.
            assert torch.equal(initial.E_bias,m.E_bias)
            results['initial_fixed_readout']=score(initial,batches)
        print(n,results[n]['accuracy_by_level'],flush=True)
    (dst/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
