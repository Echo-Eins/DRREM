"""Actual trained packet utility, bridge credit, and streaming; opened dev only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.semantic_flywheel import SemanticFlywheelMachine, FAMILIES
from drrem.core.semantic_flywheel_decode import SemanticFlywheelDecoder
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.core.causal_transport import response_objective
from drrem.data.protocol import restore_openorca_protocol, file_digest
from scripts.train_semantic_flywheel import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    ck = torch.load(a.run/'checkpoint.pt', map_location='cpu', weights_only=False)
    protocol = ck['protocol']
    m = model_from_protocol(protocol).cuda().eval(); m.load_state_dict(ck['model'])
    if not isinstance(m, SemanticFlywheelMachine):
        raise ValueError('packet diagnostics require the flywheel')
    result = dict(step=ck['step'], checkpoint_sha256=file_digest(a.run/'checkpoint.pt'),
                  seen_response_bytes=ck['seen_response_bytes'], test_opened=False,
                  scope='16 opened dev documents; lesion is not independent semantic validation')
    del ck
    data = restore_openorca_protocol(protocol['data'])
    ids = np.asarray(protocol['data']['dev_evaluated_ids'][:16])
    batches = [data.make_batch(ids[i:i+2]) for i in range(0, len(ids), 2)]
    device = torch.device('cuda')
    baseline = evaluate(m, batches, device, protocol['precision'])
    result['baseline'] = baseline
    result['family_lesions'] = {}
    for family in FAMILIES:
        m.packet_family_gains[family] = 0.
        score = evaluate(m, batches, device, protocol['precision'])
        result['family_lesions'][family] = dict(bpb=score['bpb_h1'], difference_bpb=score['bpb_h1']-baseline['bpb_h1'],
                                                documents=score['documents'])
        m.packet_family_gains[family] = 1.
    # Measure the FINAL objective's bridge, excluding the auxiliary first loss.
    b = data.make_batch(ids[:1]).to(device)
    m.train()  # enables activation recomputation only; no running statistics/dropout
    with torch.autocast('cuda', dtype=torch.bfloat16):
        final, first = m(b.x[:, :-1], b.active[:, :-1], return_first=True)
        loss, _, _ = response_objective(final, b.x, b.loss_mask[:, :-1], b.active[:, :-1], protocol['mtp_weight'])
    bridge = torch.autograd.grad(loss, first, allow_unused=True)[0]
    result['final_loss_to_first_logits_gradient_norm'] = float(bridge.float().norm()) if bridge is not None else 0.
    result['feedback_gains'] = [reader.gain.detach().tolist() for reader in m.readers]
    del final, first, loss, bridge
    m.eval()
    # Complete valid prompt plus the first 16 observed response bytes.
    start = int(torch.nonzero(b.active[0], as_tuple=False)[0])
    end = min(b.x.shape[1]-1, b.P+16)
    raw = b.x[:1, start:end]
    prefix = max(1, raw.shape[1]-16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = m(raw).float()
        changed = raw.clone(); changed[:, prefix:] = (changed[:, prefix:]+1) % 256
        past = m(changed)[:, :prefix].float()
    result['future_invariance_max_error'] = float((expected[:, :prefix]-past).abs().max())
    decoder = SemanticFlywheelDecoder(m, capacity=raw.shape[1], precision='bf16')
    pieces = [decoder.prefill(raw[:, :prefix])]
    for t in range(prefix, raw.shape[1]):
        pieces.append(decoder.step(raw[:, t]))
    streamed = torch.cat(pieces, 1).float()
    lp, lq = expected.log_softmax(-1), streamed.log_softmax(-1)
    result['streaming'] = dict(length=raw.shape[1], max_logit_error=float((expected-streamed).abs().max()),
                               mean_kl_nats=float((lp.exp()*(lp-lq)).sum(-1).mean()))
    if result['future_invariance_max_error'] != 0 or result['streaming']['mean_kl_nats'] > .002:
        raise RuntimeError('trained flywheel causal/streaming acceptance failed')
    if protocol['semantic_flywheel']['signal'] == 'live' and result['final_loss_to_first_logits_gradient_norm'] <= 0:
        raise RuntimeError('trained bridge has no gradient')
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline', 'family_lesions')}), flush=True)


if __name__ == '__main__':
    main()
