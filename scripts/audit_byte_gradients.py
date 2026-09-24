"""Same forward values, cut versus full temporal derivatives; MTP and constraints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.learning2 import doc_end, run_prompt2
from drrem.core.machine2 import MachineV2Config, make_targets
from drrem.data.protocol import restore_openorca_protocol, file_digest
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine
from drrem.rulers.temporal_adam import advance_graph, detach_state


def compare(a, b):
    a, b = a.flatten(), b.flatten()
    return {'norm_a': float(a.norm()), 'norm_b': float(b.norm()),
            'cosine': float(F.cosine_similarity(a, b, dim=0)),
            'relative_difference': float((a-b).norm()/b.norm().clamp_min(1e-20))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bytes', type=int, default=16)
    p.add_argument('--docs', type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta, saved = ck['meta'], ck['trainer']
    data = restore_openorca_protocol(meta['data'])
    order = meta['data']['response_budget']['order'][saved['batches']*meta['data']['batch']:]
    ids = np.asarray([i for i in order if len(data.responses[i]) >= a.bytes+8][:a.docs])
    b = data.make_batch(ids).to('cuda')
    m = LastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    tr.load_state_dict(saved)
    start = run_prompt2(m, b, TWIN8)
    params = {k: v for k, v in tr.twin.params.items() if v.numel()}
    end = doc_end(b)
    result = {'anchor_sha256': file_digest(a.checkpoint), 'batch': saved['batches'],
              'train_doc_ids': ids.tolist(), 'unroll_bytes': a.bytes,
              'objective': 'final response position after fixed-weight rollout; no optimizer/homeostasis steps'}
    gradients, outputs = {}, {}
    for mode in ('cut_each_byte', 'full_through_window'):
        state = start.clone()
        drives = []
        W = m.W()
        for t in range(b.P-1, b.P-1+a.bytes):
            I = m.input_drive(b.x, t)
            drives.append(I)
            active = b.active[:, t]
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, I, 8, m.xbar(state), W, um, bias=m.bias(state))
            s = m.rho(x)
            if t < b.P-2+a.bytes:
                state = advance_graph(m, state, s, x, um, b.x[:, t+1], active)
                if mode == 'cut_each_byte':
                    state = detach_state(state)
        Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
        ce = m.final_terms(s, Y, V)
        h1, mtp = ce[:, 0].mean(), ce[:, 1:].sum(1).mean()/7
        total = h1+mtp
        if mode == 'cut_each_byte':
            g1 = torch.autograd.grad(h1, tuple(params.values()), retain_graph=True, allow_unused=True)
            ga = torch.autograd.grad(mtp, tuple(params.values()), retain_graph=True, allow_unused=True)
            result['h1_vs_mtp'] = {name: compare(x.detach(), y.detach())
                                   for name, x, y in zip(params, g1, ga, strict=True) if x is not None and y is not None}
        gs = torch.autograd.grad(total, [*params.values(), *drives], allow_unused=True)
        gradients[mode] = {name: g.detach() for name, g in zip(params, gs[:len(params)], strict=True)}
        result[mode] = {'h1_nats': float(h1.detach()), 'total_nats': float(total.detach()),
                        'input_gradient_norm_by_time': [0. if g is None else float(g.norm()) for g in gs[len(params):]]}
        outputs[mode] = s.detach().clone()
        print(json.dumps({'case': mode, **result[mode]}), flush=True)
    torch.testing.assert_close(outputs['cut_each_byte'], outputs['full_through_window'], rtol=0, atol=0)
    result['cut_vs_full'] = {k: compare(gradients['cut_each_byte'][k], gradients['full_through_window'][k]) for k in params}
    result['first_adam_constraint_cancellation'] = {}
    for name, sign in [('S', 1), ('A', -1)]:
        g = gradients['cut_each_byte'][name]
        tangent = .5*(g+sign*g.T)*m.mask
        raw_step = -g/(g.abs()+1e-8)
        legacy_step = .5*(raw_step+sign*raw_step.T)*m.mask
        corrected_step = -tangent/(tangent.abs()+1e-8)
        active = (m.mask != 0) & (tangent.abs() > tangent.abs().max()*1e-5)
        result['first_adam_constraint_cancellation'][name] = {
            'fraction_legal_gradient_entries_step_below_1pct': float((legacy_step[active].abs() < .01).float().mean()),
            'legacy_predicted_descent_unit_lr': -float((g*legacy_step).sum()),
            'corrected_predicted_descent_unit_lr': -float((g*corrected_step).sum()),
            'step_comparison': compare(legacy_step, corrected_step)}
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('cut_vs_full', 'h1_vs_mtp', 'first_adam_constraint_cancellation')}), flush=True)


if __name__ == '__main__':
    main()
