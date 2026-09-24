"""Long exact-span overlap with the declared OpenOrca warm-start corpus.

Audit dev and only the already opened, locked512 test documents. Remaining
test documents are not accessed. This catches copied passages, not semantic
paraphrases, and is not a reconstruction of every ancestral training event.
"""
import json
import math
from pathlib import Path

from drrem.data.fineweb import FineWebBytes, digest
from drrem.data.protocol import restore_openorca_protocol
from scripts.audit_fineweb_overlap import coverage
from scripts.train_fineweb_transport import DEFAULT_CACHE
from scripts.summarize_fineweb import paired


def main():
    root = Path('runs/fineweb_energy_20260922')
    warm_protocol = Path('runs/semantic_flywheel_20260921/baseline/protocol.json')
    protocol = json.loads(warm_protocol.read_text())['data']
    warm = restore_openorca_protocol(protocol)
    anchors = set()
    source_bytes = 0
    for doc in protocol['response_budget']['order']:
        raw = warm.prompts[doc][-protocol['prompt_max']:] + warm.responses[doc][:protocol['resp_max']]
        source_bytes += len(raw)
        anchors.update(raw[start:start+256] for start in range(0, max(0, len(raw)-255), 128))
    corpus = FineWebBytes(DEFAULT_CACHE)
    locked = json.loads((root/'test_plan.json').read_text())
    if not (root/'independent_test.json').exists():
        raise ValueError('refuse to open a new test as part of this follow-up audit')
    result = dict(scope=__doc__, source_protocol_sha256=digest(warm_protocol),
                  source_bytes=source_bytes, unique_anchors=len(anchors),
                  anchor_width=256, source_stride=128, splits={})
    for split, ids in [('dev', corpus.splits['dev']), ('already_opened_test512', locked['test_documents'])]:
        rows = []
        for doc in ids:
            raw = corpus.document(int(doc)).tobytes()
            covered, hits = coverage(raw, anchors)
            rows.append(dict(id=int(doc), bytes=len(raw), matched_bytes=covered, anchor_hits=hits))
        total = sum(r['bytes'] for r in rows)
        matched = sum(r['matched_bytes'] for r in rows)
        result['splits'][split] = dict(documents=len(rows), bytes=total, matched_bytes=matched,
                                      matched_byte_fraction=matched/total, rows=rows)
    # Also audit the exact selected FineWeb training bytes against the locked
    # test. Filtering uses source-text overlap, never candidate/control loss.
    fineweb_plan = json.loads((root/'base8/protocol.json').read_text())['train']
    fineweb_anchors = set()
    for doc in fineweb_plan['documents']:
        raw = corpus.document(int(doc))[:fineweb_plan['response_caps'][str(doc)]].tobytes()
        fineweb_anchors.update(raw[start:start+256] for start in range(0, max(0, len(raw)-255), 128))
    union = anchors | fineweb_anchors
    rows = []
    for doc in locked['test_documents']:
        raw = corpus.document(int(doc)).tobytes()
        covered, hits = coverage(raw, union)
        rows.append(dict(id=int(doc), bytes=len(raw), matched_bytes=covered, anchor_hits=hits))
    total = sum(r['bytes'] for r in rows)
    matched = sum(r['matched_bytes'] for r in rows)
    result['splits']['test512_vs_both_training_corpora'] = dict(
        documents=len(rows), bytes=total, matched_bytes=matched,
        matched_byte_fraction=matched/total, rows=rows)
    excluded = {r['id'] for r in rows if r['matched_bytes']}
    test = json.loads((root/'independent_test.json').read_text())
    filtered = {}
    for name, entry in test['models'].items():
        kept = [row for row in entry['evaluation']['documents'] if row['id'] not in excluded]
        filtered[name] = kept
    control, candidate = locked['models']
    result['posthoc_overlap_sensitivity'] = dict(
        scope='secondary robustness check of the two already frozen endpoints; full test remains the prespecified primary result',
        excluded_document_ids=sorted(excluded),
        comparison=paired(filtered[candidate], filtered[control]),
        evaluations={name:dict(documents=len(values), raw_bytes=sum(v['bytes'] for v in values),
                              bpb=sum(v['nats'] for v in values)/sum(v['bytes'] for v in values)/math.log(2))
                     for name,values in filtered.items()})
    (root/'warm_exact_span_overlap.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: {k:v for k,v in value.items() if k != 'rows'}
                      for name,value in result['splits'].items()}), flush=True)


if __name__ == '__main__':
    main()
