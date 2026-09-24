"""Full-width training-graph and streaming acceptance before a continuation."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.semantic_flywheel import SemanticFlywheelMachine
from drrem.core.semantic_flywheel_decode import SemanticFlywheelDecoder
from drrem.data.protocol import restore_openorca_protocol, file_digest
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_semantic_flywheel import DEFAULT_PARENT, TrainingExecution, objective, SOURCES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--compile-parts', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False)
    protocol = ck['protocol']
    cfg = replace(CausalTransportConfig(**protocol['model']), checkpoint_hops=False)
    torch.manual_seed(881)
    m = SemanticFlywheelMachine(cfg).cuda()
    m.load_state_dict(ck['model'], strict=False)
    base = CausalTransportMachine(cfg).cuda().eval(); base.load_state_dict(ck['model'])
    data = restore_openorca_protocol(protocol['data'])
    ids = np.asarray(protocol['data']['response_budget']['order'][:2])
    b = pad_transport_batch(data.make_batch(ids), protocol['data']['prompt_max']+protocol['data']['resp_max'], 2).to('cuda')
    del ck
    result = dict(parent=str(a.parent.resolve()), parent_sha256=file_digest(a.parent),
                  parameters=sum(v.numel() for v in m.parameters()), slots_per_level=len(m.slot_families),
                  compile_parts=a.compile_parts, training_probe=[], optimizer_steps=0, test_opened=False,
                  probe_scope='two training documents; no weight updates; not a quality score')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        m.eval()
        final, first = m(b.x[:, :-1], b.active[:, :-1], return_first=True)
        original = base(b.x[:, :-1], b.active[:, :-1])
        error = float((first-original).abs().max())
        if error != 0:
            raise RuntimeError('first solve changed pretrained inference: ' + str(error))
        result['first_solve_base_max_error'] = error
        _, s, f, c = objective(final, first, b, protocol['mtp_weight'], .25)
        result['initial_training_slice_bpb'] = dict(first=float(f[0]/c[0])/math.log(2), second=float(s[0]/c[0])/math.log(2))
    del base, final, first, original
    execution = TrainingExecution(m, a.compile_parts)
    for iteration in range(3):
        m.train(); m.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
        with execution.active() as forward:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                final, first = forward(b.x[:, :-1], b.active[:, :-1], return_first=True)
                loss, s, f, c = objective(final, first, b, protocol['mtp_weight'], .25)
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        torch.cuda.synchronize()
        row = dict(iteration=iteration, seconds=time.perf_counter()-start,
                   peak_gib=torch.cuda.max_memory_allocated()/2**30, gradient_norm=float(norm),
                   final_bpb=float(s[0]/c[0])/math.log(2), first_bpb=float(f[0]/c[0])/math.log(2),
                   all_spatial_gradients_nonzero=all(v.weight.grad.norm()>0 for v in m.edges.values()))
        row['all_spatial_gradients_nonzero'] = bool(row['all_spatial_gradients_nonzero'])
        result['training_probe'].append(row)
        print(json.dumps(row), flush=True)
        del final, first, loss
    m.zero_grad(set_to_none=True); m.eval()
    raw = b.x[:1, :112]; valid = b.active[:1, :112]
    # Use a contiguous real, observed prefix, avoiding an all-left-padding test.
    start = int(torch.nonzero(b.active[0], as_tuple=False)[0])
    raw = b.x[:1, start:start+112]; valid = torch.ones_like(raw, dtype=torch.bool)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = m(raw, valid).float()
        changed = raw.clone(); changed[:, 96:] = (changed[:, 96:]+1) % 256
        past = m(changed, valid)[:, :96].float()
    result['future_invariance_max_error'] = float((expected[:, :96]-past).abs().max())
    if result['future_invariance_max_error'] != 0:
        raise RuntimeError('future target leaked into solve')
    decoder = SemanticFlywheelDecoder(m, capacity=112, precision='bf16')
    parts = [decoder.prefill(raw[:, :96])]
    for t in range(96, raw.shape[1]):
        parts.append(decoder.step(raw[:, t]))
    streamed = torch.cat(parts, 1).float()
    lp = expected.log_softmax(-1); lq = streamed.log_softmax(-1)
    kl = (lp.exp()*(lp-lq)).sum(-1).mean()
    result['streaming'] = dict(length=int(raw.shape[1]), prefill=96,
        max_logit_error=float((expected-streamed).abs().max()), mean_kl_nats=float(kl),
        kv_banks=len(decoder.buffers), packet_history_positions=decoder.packet_history[0][0].shape[1])
    if not torch.isfinite(kl) or kl > .002:
        raise RuntimeError('streaming divergence exceeds BF16 acceptance tolerance')
    result['source_hashes'] = {name: file_digest(name) for name in SOURCES}
    result['accepted'] = all(r['all_spatial_gradients_nonzero'] for r in result['training_probe'])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}), flush=True)


if __name__ == '__main__':
    main()
