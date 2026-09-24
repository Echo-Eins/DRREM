"""Exact long-span overlap against the ACTUAL 10 MB FineWeb training selection.

This is neither semantic deduplication nor a check of the warm OpenOrca data.
It never reads the independent test split. Every >=383 byte exact repeat
contains a 256-byte training anchor at stride128, except selection boundaries.
"""
import json
from pathlib import Path
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE


def coverage(raw,anchors,width=256):
    covered=0;end=0;hits=0
    for start in range(max(0,len(raw)-width+1)):
        if raw[start:start+width] in anchors:
            covered+=max(0,start+width-max(end,start));end=start+width;hits+=1
    return covered,hits


def main():
    root=Path('runs/fineweb_energy_20260922');corpus=FineWebBytes(DEFAULT_CACHE)
    protocol=json.loads((root/'base8/protocol.json').read_text());plan=protocol['train']
    anchors=set()
    for doc in plan['documents']:
        cap=plan['response_caps'][str(doc)]
        raw=corpus.document(int(doc))[:cap].tobytes()
        anchors.update(raw[s:s+256] for s in range(0,max(0,len(raw)-255),128))
    rows=[]
    for doc in corpus.splits['dev']:
        raw=corpus.document(int(doc)).tobytes();covered,hits=coverage(raw,anchors)
        rows.append(dict(document=int(doc),bytes=len(raw),matched_bytes=covered,anchor_hits=hits))
    result=dict(scope=__doc__,train_plan_sha256=digest(root/'base8/protocol.json'),
        train_raw_bytes=plan['raw_bytes'] if 'raw_bytes' in plan else plan.get('budget'),
        unique_anchors=len(anchors),dev_documents=len(rows),dev_bytes=sum(r['bytes'] for r in rows),
        matched_bytes=sum(r['matched_bytes'] for r in rows),rows=rows,test_read=False)
    result['matched_byte_fraction']=result['matched_bytes']/result['dev_bytes']
    (root/'dev_exact_span_overlap.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']}))


if __name__=='__main__':main()
