"""Document-isolated dense byte LM windows, with explicit coverage and EOS.

Every selected byte is supervised once per epoch, including document tails.
One shared boundary symbol (256) acts as BOS/EOS; it is learned but excluded
from byte-perplexity and the raw-byte budget. No transition joins documents.
Left context is bounded and explicit; gradients span the complete window.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from drrem.data.openorca import Batch

BOUNDARY=256
DEFAULT_SOURCE=Path('/home/echoens/Coding/Python/Mythos_P/data/fineweb_100k')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def build_cache(source,cache,seed=220922,dev_docs=512,test_docs=2048):
    source,cache=Path(source),Path(cache)
    state=json.loads((source/'state.json').read_text())
    files=[source/f['filename'] for f in state['_data_files']]
    fingerprints={p.name:digest(p) for p in files}
    if (cache/'manifest.json').exists():
        m=json.loads((cache/'manifest.json').read_text())
        if (m['source_hashes'],m['seed'],m['dev_docs'],m['test_docs'])!=(fingerprints,seed,dev_docs,test_docs):
            raise ValueError('cached corpus identity/split differs')
        return m
    cache.mkdir(parents=True,exist_ok=True)
    seen={};offsets=[0];source_rows=[];duplicates=0;row=0
    with (cache/'text.bin').open('wb') as output:
        for file in files:
            with pa.memory_map(str(file),'r') as stream:
                reader=pa.ipc.open_stream(stream)
                for batch in reader:
                    for text in batch.column(batch.schema.get_field_index('text')).to_pylist():
                        raw=text.encode('utf-8');key=hashlib.sha256(raw).digest()
                        if not raw or key in seen:
                            duplicates+=1;row+=1;continue
                        seen[key]=len(source_rows);source_rows.append(row);row+=1
                        output.write(raw);offsets.append(offsets[-1]+len(raw))
    n=len(source_rows)
    if dev_docs+test_docs>=n:raise ValueError('splits leave no training data')
    permutation=np.random.default_rng(seed).permutation(n)
    np.save(cache/'offsets.npy',np.asarray(offsets,dtype=np.int64))
    np.savez(cache/'splits.npz',dev=permutation[:dev_docs],test=permutation[dev_docs:dev_docs+test_docs],train=permutation[dev_docs+test_docs:])
    np.save(cache/'source_rows.npy',np.asarray(source_rows,dtype=np.int64))
    result=dict(source=str(source.resolve()),source_hashes=fingerprints,seed=seed,dev_docs=dev_docs,test_docs=test_docs,
                original_rows=row,documents=n,duplicates_or_empty_removed=duplicates,text_bytes=offsets[-1],
                boundary_id=BOUNDARY,vocab=257,split_policy='exact UTF8 duplicates removed BEFORE document split')
    result['cache_hashes']={f:digest(cache/f) for f in ['text.bin','offsets.npy','splits.npz','source_rows.npy']}
    (cache/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result


class FineWebBytes:
    def __init__(self,cache):
        self.cache=Path(cache);self.manifest=json.loads((self.cache/'manifest.json').read_text())
        self.text=np.memmap(self.cache/'text.bin',mode='r',dtype=np.uint8)
        self.offsets=np.load(self.cache/'offsets.npy',mmap_mode='r')
        self.splits=dict(np.load(self.cache/'splits.npz'))

    def document(self,doc):
        return self.text[self.offsets[doc]:self.offsets[doc+1]]

    def plan(self,split='train',budget=10_000_000,block=512,context=512,max_docs=None):
        if min(budget,block,context)<1:raise ValueError('positive dimensions required')
        units=[];remaining=budget;documents=[];caps={};boundary_targets=0
        ids=self.splits[split] if max_docs is None else self.splits[split][:max_docs]
        for raw_id in ids:
            doc=int(raw_id);length=len(self.document(doc));take=min(length,remaining)
            if not take:break
            complete=take==length;ntargets=take+int(complete)
            documents.append(doc);caps[str(doc)]=take;remaining-=take;boundary_targets+=int(complete)
            # start/count index the target stream bytes + optional terminal EOS.
            for start in range(0,ntargets,block):units.append((doc,start,min(block,ntargets-start),take,int(complete)))
            if not remaining:break
        return dict(split=split,requested_raw_bytes=budget,raw_bytes=budget-remaining,boundary_targets=boundary_targets,
                    block=block,context=context,documents=documents,response_caps=caps,units=units,
                    scope='all selected bytes supervised once; EOS only for complete documents; bounded causal left context')


def window_batch(corpus,plan,unit_ids,pad=True):
    units=[plan['units'][int(i)] for i in unit_ids];pairs=[]
    if not units:raise ValueError('empty batch')
    for doc,start,count,cap,complete in units:
        raw=corpus.document(doc)
        if not(0<=start<cap+complete and 0<count<=plan['block'] and start+count<=cap+complete):
            raise ValueError('invalid supervised unit')
        # The current byte immediately before a target is always available.
        left=max(0,start+1-plan['context'])
        context=(np.concatenate(([BOUNDARY],np.asarray(raw[:start],dtype=np.int64))) if left==0
                 else np.asarray(raw[left-1:start],dtype=np.int64))
        target=np.asarray(raw[start:min(start+count,cap)],dtype=np.int64)
        if start+count>cap:target=np.concatenate((target,[BOUNDARY]))
        assert len(context)>=1 and len(context)<=plan['context'] and len(target)==count
        pairs.append((context,target))
    p=plan['context'] if pad else max(len(x) for x,_ in pairs)
    r=plan['block'] if pad else max(len(y) for _,y in pairs)
    x=np.zeros((len(units),p+r),dtype=np.int64);valid=np.zeros_like(x,dtype=bool);mask=valid.copy()
    for i,(context,target) in enumerate(pairs):
        begin=p-len(context);end=p+len(target)
        x[i,begin:p]=context;x[i,p:end]=target;valid[i,begin:end-1]=True;mask[i,p-1:end-1]=True
    return Batch(torch.from_numpy(x),torch.from_numpy(mask),torch.from_numpy(valid),p,np.asarray([u[0] for u in units]))


def limit_left_context(batch,limits):
    """Vary available history per row without changing any supervised target.

    A larger common padded frame allows short/long contexts in one optimizer
    batch. This masks only old prefix positions; it never truncates response
    targets or the immediately preceding input needed to predict the first.
    """
    limits=torch.as_tensor(limits,device=batch.x.device,dtype=torch.long)
    if limits.shape!=(len(batch.doc_ids),) or bool(((limits<1)|(limits>batch.P)).any()):
        raise ValueError('one positive context limit no larger than P per row')
    keep=torch.arange(batch.x.shape[1],device=batch.x.device)[None]>=batch.P-limits[:,None]
    return Batch(batch.x.masked_fill(~keep,0),batch.loss_mask.clone(),batch.active&keep,batch.P,batch.doc_ids.copy())
