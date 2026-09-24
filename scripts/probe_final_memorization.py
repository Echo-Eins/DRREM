"""Can the final machine fit eight fixed training documents? Ordinary Adam.

Same warm moments and byte-wise updates as the actual completed run. Frozen
parameter groups are restored after structural projection/scaling as well.
Homeostasis and error-feedback remain active in every arm; a head-only arm
therefore freezes learned body tensors, not all numerical state dynamics.
This is a capacity/optimization diagnostic, never a held-out language score.
"""
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import time

import torch

from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import evaluate_bytes
from scripts.final_probe_common import setup, seen_ids, protocol, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=32)
    p.add_argument('--arms', nargs='+', choices=['all','head_only','body_only'], default=['all','head_only','body_only'])
    a = p.parse_args()
    results = {}
    for arm in a.arms:
        ck, m, tr, data = setup(a.checkpoint)
        ids = seen_ids(ck, 8, 64)
        data.cfg = replace(data.cfg, resp_max=64)
        batch = data.make_batch(ids)
        frozen = {k:v.detach().clone() for k,v in tr.twin.params.items()
                  if v.numel() and (arm == 'head_only' and not k.startswith('E_r') or arm == 'body_only' and k.startswith('E_r'))}
        def suppress(opt, args, kwargs):
            for name in frozen:
                tr.twin.params[name].grad = None
        hook = tr.twin.opt.register_step_pre_hook(suppress)
        original_scaling = m.synaptic_scaling
        @torch.no_grad()
        def scaling_and_freeze():
            result = original_scaling()
            for name, value in frozen.items():
                tr.twin.params[name].copy_(value)
            return result
        m.synaptic_scaling = scaling_and_freeze
        rows = [{'epoch':0,'fit_h1':evaluate_bytes(m,[batch],TWIN8)['bpb_h1']}]
        print({'arm':arm,**rows[-1]},flush=True)
        for epoch in range(1,a.epochs+1):
            torch.cuda.synchronize()
            start=time.perf_counter()
            info=tr.train_batch(batch)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-start
            row={'epoch':epoch,'seconds':seconds,'online_h1':info['train_h1_bpb']}
            if epoch % 4 == 0 or epoch == a.epochs:
                row['fit_h1']=evaluate_bytes(m,[batch],TWIN8)['bpb_h1']
                for name,value in frozen.items():
                    assert torch.equal(tr.twin.params[name],value),name
            rows.append(row)
            results[arm]={'protocol':protocol(a.checkpoint,ids,[]),'rows':rows,
                          'optimizer':'original warm torch.optim.Adam, original byte-wise update and homeostasis',
                          'scope':'memorization of eight previously seen training response prefixes; NOT dev/test'}
            write(a.out,results)
            if 'fit_h1' in row:
                print({'arm':arm,**row},flush=True)
                if row['fit_h1'] < .7:
                    break
        hook.remove()
        del tr,m,data,frozen
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
