"""Measure activity diversity, recurrent DC drive and the post-fit error mismatch."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import MachineV2Config, make_targets
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine


def moments(x):
    x = x.double()
    mean = x.mean(0)
    centered = x-mean
    covariance = centered.T@centered/len(x)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return {'mean_rms': float(mean.square().mean().sqrt()),
            'centered_rms': float(centered.square().mean().sqrt()),
            'dc_fraction_squared_norm': float(mean.square().sum()/x.square().sum(1).mean()),
            'covariance_participation_rank': float(eig.sum().square()/eig.square().sum().clamp_min(1e-30)),
            'top10_variance_fraction': float(eig[-10:].sum()/eig.sum().clamp_min(1e-30)),
            'constant_unit_fraction': float((centered.square().mean(0) < 1e-10).double().mean()),
            'always_zero_fraction': float((x.abs().amax(0) < 1e-8).double().mean()),
            'always_one_fraction': float(((x-1).abs().amax(0) < 1e-8).double().mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta, saved = ck['meta'], ck['trainer']
    data = restore_openorca_protocol(meta['data'])
    order = meta['data']['response_budget']['order']
    lo = saved['batches']*meta['data']['batch']
    ids = np.asarray(order[lo:lo+a.docs])
    b = data.make_batch(ids).to('cuda')
    m = LastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    tr.load_state_dict(saved)
    states, pres = [], []
    result = {'anchor_sha256': file_digest(a.checkpoint), 'train_doc_ids': ids.tolist(),
              'batch': saved['batches'], 'test_opened': False}
    end = doc_end(b)
    with torch.no_grad():
        state = run_prompt2(m, b, TWIN8)
        sampled_states = {}
        W = m.W()
        for t in range(b.P-1, b.T-1):
            if t-(b.P-1) in (0, 32, 128):
                sampled_states[t] = state.clone()
            active = b.active[:, t]
            um = m.unit_mask(state, active)
            xb = m.xbar(state)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), 8, xb, W, um, bias=m.bias(state))
            s = m.rho(x)
            _, valid = make_targets(b.x, t, m.cfg.H_max, b.P, end)
            valid = active & valid[:, 0]
            states.append(s[valid])
            pres.append((s[:, None]+xb)[valid])
            advance(m, state, s, x, um, b.x[:, t+1], active, False)
        activations, pre = torch.cat(states), torch.cat(pres)
        result['activation_by_level'] = [moments(v) for v in activations.split(m.cfg.N, 1)]
        result['pre_by_target_level'] = [moments(pre[:, l]) for l in range(m.cfg.L)]
    del states, pres, activations, pre
    result['real_adam_steps'] = {}
    for t, state in sampled_states.items():
        tr.load_state_dict(saved)
        result['real_adam_steps'][str(t-(b.P-1))] = probe_step(m, tr, b, state, t, end)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


def probe_step(m, tr, b, state, t, end):
    active = b.active[:, t]
    um = m.unit_mask(state, active)
    W = m.W()
    xb = m.xbar(state)
    x, _ = m.run_free(state.x, m.input_drive(b.x, t), 8, xb, W, um, bias=m.bias(state))
    s = m.rho(x)
    Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
    valid = active & V[:, 0]
    loss = m.loss_per_sample(s, Y, V, state.tick)[valid].mean()
    with torch.no_grad():
        old_p = m.probs_h1(s)
        old_force = m.h1_error_force(s, b.x[:, t+1])
        old_h1 = m.final_terms(s, Y, V)[valid, 0].mean()
        old_W = W.detach().clone()
    tr.twin.opt.zero_grad(set_to_none=True)
    loss.backward()
    tr.twin.opt.step()
    tr.twin.project()
    m.synaptic_scaling()
    with torch.no_grad():
        new_p = m.probs_h1(s)
        new_force = m.h1_error_force(s, b.x[:, t+1])
        result = {
            'optimizer_moments_reset': False, 'valid_docs': int(valid.sum()),
            'h1_before_at_same_state_bits': float(old_h1)/math.log(2),
            'h1_after_at_same_state_bits': float(m.final_terms(s, Y, V)[valid, 0].mean())/math.log(2),
            'prediction_kl_bits': float((old_p*(old_p.clamp_min(1e-30).log()-new_p.clamp_min(1e-30).log())).sum(1).mean())/math.log(2),
            'error_force_relative_change': float((old_force-new_force).norm()/old_force.norm()),
            'error_force_cosine': float(torch.nn.functional.cosine_similarity(old_force.flatten(), new_force.flatten(), dim=0))}
        shift = m.recurrent_drive(s, xb, m.W()-old_W)[valid]
        result['recurrent_field_change_by_level'] = [
            {k: v for k, v in moments(z).items() if k in ('dc_fraction_squared_norm', 'mean_rms', 'centered_rms')}
            for z in shift.split(m.cfg.N, 1)]
    return result


if __name__ == '__main__':
    main()
