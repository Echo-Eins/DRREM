"""Can the first contextual top-level result return through the bottom level?

Patch only level0 after hop5 with its counterfactual state after perturbing
the first nonzero top temporal read (hop3). Other levels are kept original.
The final decoder cannot see this patch at H6; H7 is the first reachable time.
This is a reachability test, not a quality claim for adding an untrained hop.
"""
from dataclasses import replace
import json
from pathlib import Path
import torch
from drrem.diagnostics.functional_map import module_scales
from scripts.audit_transport_functions import BASE,load,save
from drrem.data.protocol import restore_openorca_protocol


def patch_after(model,ids,valid,patch,at=5,level=0):
    original=model.transport_hop;counter=0
    def hop(*args):
        nonlocal counter
        out=original(*args);counter+=1
        if counter==at:out=tuple(patch if i==level else x for i,x in enumerate(out))
        return out
    model.transport_hop=hop
    try:return model(ids,valid)
    finally:model.transport_hop=original


def probe(model,ids,valid):
    cfg=model.cfg;rows=[]
    for hops in [6,7]:
        # Keep original step_scale: changing it would confound timing.
        model.cfg=replace(cfg,hops=hops)
        with torch.no_grad():
            base=model(ids,valid)
            _,a=model.forward_states(ids,valid,return_hops=True)
            with module_scales(model,{'temporal.2@2':.5}):
                _,b=model.forward_states(ids,valid,return_hops=True)
            patched=patch_after(model,ids,valid,b[5][0])
        states,path=model.forward_states(ids,valid,return_hops=True)
        out=torch.einsum('btn,hvn->bthv',model.final_norm(states[-1]),model.readout)
        derivative=torch.autograd.grad(out.square().mean(),path[5][0],allow_unused=True)[0]
        rows.append(dict(hops=hops,step_scale=model.step_scale,
            bottom_change_after_hop4=float((a[4][0]-b[4][0]).abs().max()),
            bottom_change_after_hop5=float((a[5][0]-b[5][0]).abs().max()),
            logits_change_via_bottom_patch=float((patched-base).abs().max()),
            final_derivative_to_bottom_after_hop5=0. if derivative is None else float(derivative.norm())))
    model.cfg=cfg
    return rows


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    m,ck=load(BASE);data=restore_openorca_protocol(ck['protocol']['data'])
    raw=data.prompts[ck['protocol']['data']['response_budget']['order'][137]][-64:]
    ids=torch.tensor(list(raw),device='cuda')[None];valid=torch.ones_like(ids,dtype=torch.bool)
    rows=probe(m,ids,valid)
    assert rows[0]['bottom_change_after_hop4']==0 and rows[0]['bottom_change_after_hop5']>0
    assert rows[0]['logits_change_via_bottom_patch']==0 and rows[0]['final_derivative_to_bottom_after_hop5']==0
    assert rows[1]['final_derivative_to_bottom_after_hop5']>0
    result=dict(scope='real1024x3 trained checkpoint,FP32,one TRAIN prefix; timing proof,not H7 language improvement',
                minimum_hops_for_top_context_bottom_top='(L-1)+1+2*(L-1)=3L-2 for synchronous zero-initialized upper levels',rows=rows)
    save(Path('runs/functional_map_20260922/feedback_roundtrip.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
