"""Exact final-loss sensitivity at every level and hop on fixed prefix cases.

Each state's adjoint is its actual partial derivative in the original graph.
Only the last-position slice is measured, so it has no credit from later
positions. Dead late updates are reported as unconsumed, not tiny gradients.
"""
import argparse
import json
from pathlib import Path

import torch

from drrem.data.fineweb import FineWebBytes, digest
from drrem.diagnostics.consumer_replay import decode, rms, cosine
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.probe_ff_credit import batch, losses


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cases',type=int,default=256);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    protocol=json.loads((a.audit/'protocol.json').read_text());parent=Path(protocol['checkpoint'])
    if digest(parent)!=protocol['checkpoint_sha256']:raise ValueError('checkpoint changed')
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model']);m.requires_grad_(False)
    encoder=m.encode_input
    m.encode_input=lambda ids,valid:encoder(ids,valid).detach().requires_grad_(True)
    after=m.after_hop
    def live_constants(states,hop):
        # Initial upper-level constants still admit counterfactual state
        # interventions. Make those leaves differentiable without changing
        # values or severing any existing dependency.
        return tuple(s if s.requires_grad else s.requires_grad_(True) for s in after(states,hop))
    m.after_hop=live_constants
    corpus=FineWebBytes(DEFAULT_CACHE)
    all_cases=protocol['held_cases'];rows=all_cases[::max(1,len(all_cases)//a.cases)][:a.cases]
    trace={(h,l):[] for h in range(1,m.cfg.hops+1) for l in range(m.cfg.layers)}
    for start in range(0,len(rows),8):
        ids,valid,y=batch(corpus,rows[start:start+8],protocol['arguments']['length'],'cuda')
        end,trajectory=m.forward_states(ids,valid,True)
        logits=decode(m,end,ids,valid);h1,aux=losses(logits,y)
        ordered=[s for states in trajectory[1:] for s in states]
        gradients=torch.autograd.grad((h1+aux).sum(),ordered,allow_unused=True)
        for (hop,level),g in zip(trace,gradients):
            z=trajectory[hop][level].detach()[:,-1]
            delta=z-trajectory[hop-1][level].detach()[:,-1]
            if g is None:
                values=torch.zeros(len(z),4); values[:,3]=1
                values[:,0]=rms(z)[:,0].cpu()
            else:
                grad=g.detach()[:,-1]
                # Predicted decrease under a 3%-RMS directed state change.
                sensitivity=.03*rms(z)[:,0]*rms(grad)[:,0]*m.cfg.neurons
                values=torch.stack([rms(z)[:,0],sensitivity,cosine(-grad,delta),torch.zeros_like(sensitivity)],-1).cpu()
            trace[hop,level].append(values)
        del logits,trajectory,end,gradients,ordered
    result=dict(scope=__doc__,checkpoint_sha256=protocol['checkpoint_sha256'],cases=rows,
                test_opened=False,source_sha256=digest(__file__),precision='FP32',levels=[])
    for (hop,level),parts in trace.items():
        z=torch.cat(parts)
        result['levels'].append(dict(hop=hop,level=level,state_rms=float(z[:,0].mean()),
            predicted_joint_decrease_nats_at_3pct=float(z[:,1].mean()),
            incoming_step_descent_cosine=float(z[:,2].mean()),unused_fraction=float(z[:,3].mean())))
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['levels']),flush=True)


if __name__=='__main__':main()
