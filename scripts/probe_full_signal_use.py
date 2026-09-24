"""Read-only lesions of the actual 3 MB checkpoints, using their frozen code.

Removing a learned branch tests its use, not whether training that branch is
better than training the control architecture. All results are dev only.
"""
import argparse
import json
from pathlib import Path
import sys
import types


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root / 'source'))
    import torch
    from torch.nn import functional as F
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.fineweb import FineWebBytes, window_batch, digest
    from scripts.train_full_signal_trial import make_trial_model, evaluate_horizons
    from scripts.train_fineweb_transport import DEFAULT_CACHE
    from scripts.summarize_fineweb import paired

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    corpus = FineWebBytes(DEFAULT_CACHE)
    result = dict(scope=__doc__, test_opened=False, models={})
    for name in ('warm_fourier', 'fresh_fourier', 'warm_bridge', 'warm_fourier_bridge'):
        folder = a.root / name
        path = folder / 'checkpoint_3mb.pt'
        ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if ck['raw_byte_exposures'] != 3_001_924:
            raise ValueError('lesion checkpoint is not the declared 3 MB endpoint')
        protocol = ck['protocol']
        for file, expected in protocol['source_hashes'].items():
            if digest(a.root / 'source' / file) != expected:
                raise ValueError('frozen source changed')
        model = make_trial_model(protocol['variant'], CausalTransportConfig(**protocol['model'])).cuda().eval()
        model.load_state_dict(ck['model'])
        saved = [json.loads(s) for s in (folder / 'metrics_3mb.jsonl').read_text().splitlines()]
        saved_dev = next(r['dev'] for r in reversed(saved) if 'dev' in r)
        reference = evaluate_horizons(model, corpus, protocol['dev'])
        if abs(reference['bpb'] - saved_dev['bpb']) > 1e-7:
            raise ValueError('native checkpoint evaluation did not reproduce the saved score')
        # Keep valid positions fixed; edit only future bytes in an actual window.
        b = window_batch(corpus, protocol['dev'], [0]).to('cuda')
        ids, valid = b.x[:, :-1], b.active[:, :-1]
        cut = b.P + 64
        altered = ids.clone()
        altered[:, cut:] = (altered[:, cut:] + 1) % model.cfg.vocab
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            left, right = model(ids, valid), model(altered, valid)
        if not torch.equal(left[:, :cut], right[:, :cut]):
            raise ValueError('future edit changed earlier predictions')
        del left, right, b, ids, valid, altered
        base = dict(checkpoint_sha256=digest(path), native=reference, future_invariant=True, lesions={})
        result['models'][name] = base
        modes = []
        fourier = [edge for edge in model.edges.values() if hasattr(edge, 'coefficients')]
        if fourier:
            modes.extend(['edge_functions_off', 'edge_functions_linearized', 'frequencies_reset'])
        if hasattr(model, 'bridge_gain'):
            modes.append('bridges_off')
        for mode in modes:
            original_functions = []
            try:
                with torch.no_grad():
                    if mode == 'edge_functions_off':
                        for edge in fourier:
                            edge.coefficients.zero_()
                    elif mode == 'edge_functions_linearized':
                        # Same edge matrices and source frequencies; replace
                        # sin(phi) by phi and cos(phi)-1 by zero at every hop.
                        def linearized(edge, x):
                            phase = x * F.softplus(edge.raw_frequency).to(x.dtype)
                            return phase, torch.zeros_like(phase)
                        for edge in fourier:
                            original_functions.append((edge, edge.functions))
                            edge.functions = types.MethodType(linearized, edge)
                    elif mode == 'frequencies_reset':
                        import math
                        for edge in fourier:
                            edge.raw_frequency.fill_(math.log(math.expm1(1.)))
                    elif mode == 'bridges_off':
                        for gain in model.bridge_gain.values():
                            gain.zero_()
                score = evaluate_horizons(model, corpus, protocol['dev'])
                base['lesions'][mode] = dict(evaluation=score, vs_native=paired(score['documents'], reference['documents']))
                print(json.dumps(dict(model=name, lesion=mode, bpb=score['bpb'],
                                      vs_native=base['lesions'][mode]['vs_native'])), flush=True)
                a.out.write_text(json.dumps(result, indent=2) + '\n')
            finally:
                for edge, function in original_functions:
                    edge.functions = function
                model.load_state_dict(ck['model'])
        # Lesions must not silently leak into the next model or a saved file.
        with torch.no_grad():
            for parameter_name, q in model.named_parameters():
                if not torch.equal(q.cpu(), ck['model'][parameter_name]):
                    raise ValueError('lesion was not restored')
        del ck, model, fourier, original_functions
        torch.cuda.empty_cache()
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
