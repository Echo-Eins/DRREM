"""Read-only context truncation at identical scored response positions.

Reconstruct every recurrent state channel from the last K observed bytes and
compare with the complete causal prefix. Targets never enter the scored state.
This measures dependence on history, not addressable or semantic memory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.learning2 import advance, doc_end
from drrem.core.machine2 import MachineV2Config, make_targets
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer
from drrem.rulers.temporal_adam import ByteChunkAdam


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=64)
    p.add_argument('--positions', type=int, nargs='+', default=[0, 16, 64, 128])
    p.add_argument('--contexts', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved = ck['trainer']
    meta = ck.get('meta') or ck['protocol']['anchor_protocol']
    data = restore_openorca_protocol(meta['data'])
    ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.docs])
    b = data.make_batch(ids).to('cuda')
    centered = 'field_bias' in saved['machine']
    cls = CenteredLastDecoderMachine if centered else LastDecoderMachine
    m = cls(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    rates = dict(lr=meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    trainer = (ByteChunkAdam(m, TWIN8, **rates, **saved['credit_config']) if centered else ByteAdam(m, TWIN8, **rates))
    if centered:
        attach_field_optimizer(trainer)
    trainer.load_state_dict(saved)
    end = doc_end(b)
    result = {'checkpoint_sha256': file_digest(a.checkpoint), 'dev_ids': ids.tolist(),
              'test_opened': False, 'definition': 'K observed bytes, including the current input byte',
              'positions': {}}

    starts = {max(0, b.P-1+pos-k+1) for pos in a.positions for k in a.contexts}
    entries = {}

    @torch.no_grad()
    def run(start, stop, entry=None, capture=False):
        state = m.init_state(len(ids)) if entry is None else entry.clone()
        W = m.W()
        outputs = {}
        for t in range(start, stop+1):
            if capture and t in starts:
                entries[t] = state.clone()
            active = b.active[:, t]
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), 8, m.xbar(state), W, um, bias=m.bias(state))
            s = m.rho(x)
            if t-b.P+1 in a.positions:
                outputs[t-b.P+1] = m.probs_h1(s).clone()
            advance(m, state, s, x, um, b.x[:, t+1], active, False)
        return outputs

    full = run(0, b.P-1+max(a.positions), capture=True)
    for position in a.positions:
        t = b.P-1+position
        Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
        valid = b.active[:, t] & V[:, 0]
        pred = full[position].clamp_min(1e-30)
        truth = Y[:, :1]
        ce = -pred.gather(1, truth).log2().squeeze(1)
        row = {'count': int(valid.sum()), 'full_h1_bpb': float(ce[valid].mean()), 'contexts': []}
        for k in a.contexts:
            start = max(0, t-k+1)
            cut = run(start, t)[position].clamp_min(1e-30)
            nll = -cut.gather(1, truth).log2().squeeze(1)
            kl = (pred*(pred.log2()-cut.log2())).sum(1)
            # Keep a fully aged state, but give each document another document's
            # past. This separates cold-start damage from content dependence.
            order = torch.arange(len(ids), device=m.device)
            for previous_active in (False, True):
                eligible = valid & ((b.active[:, start-1] if start else torch.zeros_like(valid)) == previous_active)
                ix = eligible.nonzero().flatten()
                order[ix] = ix.roll(1)
            shuffled = entries[start].clone()
            for name, value in vars(shuffled).items():
                if isinstance(value, torch.Tensor):
                    setattr(shuffled, name, value[order].clone())
            if start and shuffled.err is not None:
                # The carried innovation already observed x[start]. Replace
                # only that byte, so both histories observe the same suffix.
                with torch.no_grad():
                    readout = m.E_r[-1][0, :, :m.cfg.N]
                    delta = (1-m.fly_decay)*(readout[b.x[:, start]]-readout[b.x[order, start]])/m.cfg.tau_r
                    shuffled.err[:, -m.cfg.N:] += delta*b.active[:, start-1, None]
            mixed = run(start, t, shuffled)[position].clamp_min(1e-30)
            mixed_ce = -mixed.gather(1, truth).log2().squeeze(1)
            mixed_kl = (pred*(pred.log2()-mixed.log2())).sum(1)
            if k == a.contexts[0]:
                identical = run(start, t, entries[start])[position]
                torch.testing.assert_close(identical, full[position], rtol=0, atol=0)
            row['contexts'].append({'bytes': k, 'h1_bpb': float(nll[valid].mean()),
                                    'delta_bpb': float((nll-ce)[valid].mean()),
                                    'kl_from_full_bits': float(kl[valid].mean()),
                                    'aged_other_history_delta_bpb': float((mixed_ce-ce)[valid].mean()),
                                    'aged_other_history_kl_bits': float(mixed_kl[valid].mean())})
        result['positions'][str(position)] = row
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'position': position, **row}), flush=True)


if __name__ == '__main__':
    main()
