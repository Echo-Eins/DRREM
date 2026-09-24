"""Exact value coefficients of a learned delta memory; no weight updates.

This conditions on the produced keys, queries and write strengths. It measures
the temporal reader, not the complete network's causal contribution or its
semantic accuracy. Signed coefficients are not attention probabilities.
"""
import argparse
import json
from pathlib import Path

import torch

from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import file_digest, restore_openorca_protocol


def delta_coefficients(query, keys, strength):
    """Read-before-write coefficients for B,H,S,K keys and B,H,K query.

    Reverse multiplication by (I - beta*k*k.T) gives the exact coefficient
    of each original value, including interference from subsequent writes.
    """
    residual = query
    coefficients = []
    for t in range(keys.shape[2] - 1, -1, -1):
        key = keys[:, :, t]
        coefficient = strength[:, :, t] * (residual * key).sum(-1)
        coefficients.append(coefficient)
        residual = residual - coefficient[..., None] * key
    if not coefficients:
        return keys.new_empty(*keys.shape[:2], 0)
    return torch.stack(coefficients[::-1], -1)


def coefficient_summary(coefficients, valid, query_position, prompt_boundary):
    lag = query_position - torch.arange(query_position, device=coefficients.device)
    absolute = coefficients.abs() * valid[:, None]
    mass = absolute.sum(-1).clamp_min(1e-30)
    bins = [(1, 16), (17, 64), (65, 128), (129, 256), (257, 512), (513, 4096)]
    return {
        'absolute_sum_by_head': mass.mean(0).tolist(),
        'signed_sum_by_head': coefficients.sum(-1).mean(0).tolist(),
        'effective_count_by_head': (mass.square() / absolute.square().sum(-1).clamp_min(1e-30)).mean(0).tolist(),
        'negative_absolute_fraction_by_head': ((absolute * (coefficients < 0)).sum(-1) / mass).mean(0).tolist(),
        'prompt_absolute_fraction_by_head': (absolute[:, :, :prompt_boundary].sum(-1) / mass).mean(0).tolist(),
        'absolute_mass_fraction_by_lag': {
            f'{lo}-{hi}': (absolute[:, :, (lag >= lo) & (lag <= hi)].sum(-1) / mass).mean(0).tolist()
            for lo, hi in bins
        },
    }


def oracle_label_recall(keys, strength, valid, label_count=64, seed=172):
    """Capacity probe with independent random values and perfect past-key queries.

    The real model does not receive oracle queries. This only asks whether its
    produced addresses and write law can retain unrelated records at all.
    """
    from drrem.core.nondecay_transport import phase_scan
    batch, heads, length, _ = keys.shape
    labels = torch.randint(label_count, (batch, length), generator=torch.Generator().manual_seed(seed)).to(keys.device)
    values = torch.nn.functional.one_hot(labels, label_count).to(keys.dtype)[:, None].expand(-1, heads, -1, -1)
    ages = length - torch.arange(length, device=keys.device)
    writes = keys * valid[:, None, :, None]
    arms = {}
    for name, beta in [('learned_delta', strength), ('delta_half', .5 * valid[:, None].expand_as(strength)),
                       ('delta_eighth', .125 * valid[:, None].expand_as(strength)), ('sum', None)]:
        _, state = phase_scan(keys, writes, values, beta, chunk=128)
        arms[name] = (keys @ state).mean(1).argmax(-1)
    similarity = keys @ keys.transpose(-1, -2)
    softmax = (30 * similarity).masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    arms['stored_keys_softmax30'] = (softmax @ values).mean(1).argmax(-1)
    by_arm = {}
    for name, prediction in arms.items():
        correct = prediction == labels
        rows = []
        for lo, hi in [(1, 16), (17, 64), (65, 128), (129, 256), (257, 512), (513, 4096)]:
            selected = valid & (ages >= lo) & (ages <= hi)
            n = int(selected.sum())
            rows.append({'lag': [lo, hi], 'count': n, 'accuracy': float(correct[selected].float().mean()) if n else None})
        by_arm[name] = rows
    eigen = torch.linalg.eigvalsh(writes.transpose(-1, -2) @ writes).clamp_min(0)
    mass = eigen / eigen.sum(-1, keepdim=True).clamp_min(1e-30)
    rank = (-torch.special.xlogy(mass, mass).sum(-1)).exp()
    return {'classes': label_count, 'chance': 1 / label_count, 'query': 'exact originally written key, oracle',
            'value': 'independent random label, identical across heads; not actual LM values',
            'by_arm': by_arm, 'key_covariance_entropy_rank_by_head': rank.mean(0).tolist()}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', default='checkpoint.pt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--documents', type=int, default=4)
    p.add_argument('--oracle-recall', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        raise FileExistsError(a.out)
    protocol = json.loads((a.run / 'protocol.json').read_text())
    if protocol.get('adaptive_phase', {}).get('rule') != 'delta':
        raise ValueError('this probe requires the delta memory rule')
    model = model_from_protocol(protocol).to(a.device).eval()
    checkpoint = torch.load(a.run / a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    step = checkpoint['step']
    del checkpoint
    data = restore_openorca_protocol(protocol['data'])
    ids = protocol['data']['dev_evaluated_ids'][:a.documents]
    records = []
    recall = []
    for doc in ids:
        batch = data.make_batch([doc]).to(a.device)
        inputs, valid = batch.x[:, :-1], batch.active[:, :-1]
        targets = sorted(set(min(batch.P - 1 + offset, inputs.shape[1] - 1) for offset in [0, 15, 63, 127]))
        counters = [0] * model.cfg.layers
        handles = []

        def capture(level):
            def hook(module, args):
                x, context = args[:2]
                hop = counters[level]
                counters[level] += 1
                q, k, v, _ = module.features(x, context[0])
                # Match production: state algebra and write strengths in FP32.
                with torch.autocast(x.device.type, enabled=False):
                    beta = module.write_strength(x).transpose(1, 2).sigmoid() * context[0][:, None]
                    if a.oracle_recall and hop == model.cfg.hops - 1:
                        t = batch.P - 1
                        recall.append(dict(document=int(doc), hop=hop, layer=level,
                                           **oracle_label_recall(k[:, :, :t], beta[:, :, :t], valid[:, :t])))
                    for t in targets:
                        if not bool(valid[0, t]):
                            continue
                        coefficients = delta_coefficients(q[:, :, t], k[:, :, :t], beta[:, :, :t])
                        row = coefficient_summary(coefficients, valid[:, :t], t, batch.P)
                        row.update(document=int(doc), hop=hop, layer=level, position=t, response_offset=t - batch.P + 1)
                        row['mean_write_strength_by_head'] = (beta.sum(-1) / valid.sum(-1)[:, None]).mean(0).tolist()
                        # Direct coefficient reconstruction of the reader, before
                        # the learned output projection, for an independent check.
                        from drrem.core.nondecay_transport import phase_scan
                        predicted = (coefficients[..., None] * v[:, :, :t]).sum(2)
                        actual, _ = phase_scan(q[:, :, :t + 1], k[:, :, :t + 1] * valid[:, None, :t + 1, None],
                                               v[:, :, :t + 1], beta[:, :, :t + 1], chunk=module.phase_config.chunk)
                        row['value_reconstruction_max_error'] = float((actual[:, :, t] - predicted).abs().max())
                        records.append(row)
            return hook

        try:
            for level, reader in enumerate(model.temporal):
                handles.append(reader.register_forward_pre_hook(capture(level)))
            # CPU FP32 for the diagnostic, explicitly recorded below.
            model(inputs, valid)
        finally:
            for handle in handles:
                handle.remove()
        print(json.dumps({'document': int(doc), 'records': len(records)}), flush=True)
    result = {
        'scope': 'exact conditional value coefficients; not full-network lesions, not semantic quality',
        'checkpoint_sha256': file_digest(a.run / a.checkpoint), 'step': step,
        'precision': 'FP32 diagnostic', 'documents': ids,
        'no_elapsed_time_decay': True,
        'interfering_writes_can_overwrite': True,
        'oracle_capacity': recall,
        'records': records,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
