"""Order interventions on the final byte machine, with every state path intact.

Reverse earlier bytes while preserving their multiset, current input, target,
document, and aged entry state. Replay starts one byte BEFORE the intervention:
the error state already observes the next input, so starting at the changed
byte itself would carry an inconsistent innovation from the original prefix.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.core.learning2 import advance
from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer
from drrem.rulers.temporal_adam import ByteChunkAdam


@torch.no_grad()
def replay(m, batch, start, stop, state=None, save_entries=(), score_positions=()):
    state = m.init_state(len(batch.doc_ids)) if state is None else state.clone()
    W = m.W()
    entries, scored = {}, {}
    for t in range(start, stop+1):
        if t in save_entries:
            entries[t] = state.clone()
        active = batch.active[:, t]
        m.decide_ticks(state, active, adapt=False)
        um = m.unit_mask(state, active)
        record = t in score_positions or t == stop
        x, trajectory = m.run_free(state.x, m.input_drive(batch.x,t), 8,
                                   m.xbar(state), W, um, record=record, bias=m.bias(state))
        s = m.rho(x)
        if record:
            scored[t] = {'p': m.probs_h1(s).clone(), 'hops': torch.stack(trajectory),
                         's': s.clone(), 'xbar': m.xbar(state).clone(),
                         'dam_weights': [v.clone() for v in m.dam_weights(s)]}
        if t < stop:
            advance(m,state,s,x,um,batch.x[:,t+1],active,False)
    return entries, scored


def reverse_window(batch, first, end):
    if first < 0 or first >= end or not bool(batch.active[:,first:end].all()):
        raise ValueError('intervention must contain active observed bytes only')
    x = batch.x.clone()
    x[:,first:end] = x[:,first:end].flip(1)
    return replace(batch,x=x)


def difference(m, baseline, altered, target):
    p, q = baseline['p'].clamp_min(1e-30), altered['p'].clamp_min(1e-30)
    nll, nll_q = -p.gather(1,target[:,None]).log2().flatten(), -q.gather(1,target[:,None]).log2().flatten()
    # Normalize by differences BETWEEN documents, not by the common DC field.
    sh = baseline['hops'].reshape(9,-1,m.cfg.L,m.cfg.N)
    ah = altered['hops'].reshape_as(sh)
    variance = (sh-sh.mean(1,keepdim=True)).square().mean((1,3))
    delta = (ah-sh).square().mean((1,3))
    return {'baseline_h1_bits':float(nll.mean()),'altered_h1_bits':float(nll_q.mean()),
            'delta_h1_bits':float((nll_q-nll).mean()),
            'per_document_delta_bits':(nll_q-nll).tolist(),
            'prediction_kl_bits':float((p*(p.log2()-q.log2())).sum(1).mean()),
            'prediction_total_variation':float((p-q).abs().sum(1).mean()/2),
            'state_difference_rms_by_hop_layer':delta.sqrt().tolist(),
            'difference_over_between_document_rms':(delta/variance.clamp_min(1e-30)).sqrt().tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--docs',type=int,default=16)
    p.add_argument('--positions',type=int,nargs='+',default=[64,128])
    a=p.parse_args()
    if min(a.positions)<64: p.error('positions must permit 64 active preceding bytes')
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    saved=ck['trainer']; meta=ck['protocol']['anchor_protocol']
    data=restore_openorca_protocol(meta['data'])
    ids=np.asarray([i for i in meta['data']['dev_evaluated_ids']
                    if len(data.responses[i])>max(a.positions)][:a.docs])
    if len(ids)!=a.docs: raise ValueError('not enough eligible predefined dev documents')
    b=data.make_batch(ids).to('cuda')
    m=CenteredLastDecoderMachine(MachineV2Config(**saved['machine']['cfg']),'cuda',saved['mtp_weight'])
    tr=ByteChunkAdam(m,TWIN8,lr=3e-4,core_lr=3e-6,**saved['credit_config'])
    attach_field_optimizer(tr);tr.load_state_dict(saved)
    for v in tr.twin.params.values(): v.requires_grad_(False)
    cases={f'reverse_last_{k}':(k,0) for k in (2,4,8,16,32,64)}
    cases.update(reverse_older_keep_last8=(64,8),reverse_older_keep_last16=(64,16),
                 reverse_older_keep_last32=(64,32))
    positions=[b.P-1+v for v in a.positions]
    starts={t-k-1 for t in positions for k,_ in cases.values()}
    entries,full=replay(m,b,0,max(positions),save_entries=starts,score_positions=positions)
    c=m.c.detach().cpu()
    result={'checkpoint_sha256':file_digest(a.checkpoint),'dev_ids':ids.tolist(),'test_opened':False,
            'weights_updated':False,'intervention':'reverse only observed past bytes; preserve current input and future',
            'c_mean_by_layer_channel':c.mean(-1).tolist(),
            'delay_coefficients_nonuniform_fraction_by_layer':[
                float((v-v.mean(0,keepdim=True)).norm()/v.norm()) for v in c[:,4:]],
            'positions':{},
            'limitations':['finite dev positions, not a trained permutation task',
                           'order sensitivity alone does not establish useful memory or attention',
                           'inference intervention, not an architecture training ablation']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for offset,t in zip(a.positions,positions):
        row={'cases':{},'associative_prototypes':[]}
        for weights in full[t]['dam_weights']:
            entropy=-(weights*weights.clamp_min(1e-30).log()).sum(1)
            row['associative_prototypes'].append({'entropy_effective_count':float(entropy.exp().mean()),
                'top1_weight_mean':float(weights.max(1).values.mean()),
                'between_document_weight_rms':float((weights-weights.mean(0)).square().mean().sqrt())})
        for name,(k,keep) in cases.items():
            first,end=t-k,t-keep
            altered=reverse_window(b,first,end)
            assert torch.equal(altered.x[:,t:],b.x[:,t:])
            entry=entries[first-1]
            _,identity=replay(m,b,first-1,t,entry)
            torch.testing.assert_close(identity[t]['p'],full[t]['p'],rtol=0,atol=0)
            _,scored=replay(m,altered,first-1,t,entry)
            rec=difference(m,full[t],scored[t],b.x[:,t+1])
            rec['changed_byte_fraction']=float((altered.x[:,first:end]!=b.x[:,first:end]).float().mean())
            row['cases'][name]=rec
            print(json.dumps({'position':offset,'case':name,
                **{key:rec[key] for key in ('delta_h1_bits','prediction_kl_bits','prediction_total_variation')}}),flush=True)
        result['positions'][str(offset)]=row
        a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
