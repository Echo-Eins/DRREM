"""Causal response windows with explicit, non-overlapping supervised bytes.

Used for a subsequent training stage after the original fixed 10 MB budget.
Already supervised response prefixes are excluded from the new target stream;
they may still appear as ordinary teacher-forced context. Development and test
documents, including exact text duplicates, are excluded before making units.
"""
import hashlib

import numpy as np
import torch

from drrem.config import DataConfig
from drrem.data.openorca import Batch,OpenOrcaBytes
from drrem.data.protocol import file_digest


def document_key(prompt,response):
    return hashlib.sha256(prompt+b'\0'+response).digest()


def unseen_response_units(responses,train_ids,previous_prefixes,window_bytes=256):
    if window_bytes<1:raise ValueError('positive response window required')
    units=[]
    for doc in train_ids:
        doc=int(doc);start=int(previous_prefixes.get(str(doc),previous_prefixes.get(doc,0)))
        if not 0<=start<=len(responses[doc]):raise ValueError(f'invalid previous response coverage for document {doc}')
        while start<len(responses[doc]):
            length=min(window_bytes,len(responses[doc])-start)
            units.append((doc,start,length));start+=length
    return np.asarray(units,dtype=np.int64).reshape(-1,3)


class ResponseWindows:
    def __init__(self,data,units,prompt_max=512):
        if prompt_max<1:raise ValueError('positive context window required')
        self.data=data;self.units=np.asarray(units,dtype=np.int64).reshape(-1,3);self.prompt_max=prompt_max
        for doc,start,length in self.units:
            if not (0<=doc<len(data.responses) and start>=0 and length>0 and start+length<=len(data.responses[doc])):
                raise ValueError('response unit outside document')

    def make_batch(self,unit_indices):
        units=self.units[np.asarray(unit_indices,dtype=np.int64)]
        if not len(units):raise ValueError('empty window batch')
        contexts=[(self.data.prompts[d]+self.data.responses[d][:s])[-self.prompt_max:] for d,s,n in units]
        responses=[self.data.responses[d][s:s+n] for d,s,n in units]
        if min(map(len,contexts))<1:raise ValueError('each target needs a preceding input byte')
        p=max(map(len,contexts));r=max(map(len,responses));shape=(len(units),p+r)
        x=np.zeros(shape,dtype=np.int64);active=np.zeros(shape,dtype=bool);loss=np.zeros(shape,dtype=bool)
        for row,(context,response) in enumerate(zip(contexts,responses,strict=True)):
            first=p-len(context);end=p+len(response)
            x[row,first:p]=np.frombuffer(context,dtype=np.uint8)
            x[row,p:end]=np.frombuffer(response,dtype=np.uint8)
            active[row,first:end-1]=True;loss[row,p-1:end-1]=True
        return Batch(torch.from_numpy(x),torch.from_numpy(loss),torch.from_numpy(active),p,units[:,0].copy())


def prepare_unseen_windows(previous_data_protocol,prompt_max=512,window_bytes=256):
    """Restore FULL responses, not the response caps of the preceding stage."""
    source=previous_data_protocol['files']['parquet']
    if file_digest(source['path'])!=source['sha256']:raise ValueError('dataset bytes changed')
    data=OpenOrcaBytes(DataConfig(path=source['path'],prompt_max=prompt_max,resp_max=window_bytes))
    if data.source_row.tolist()!=previous_data_protocol['source_ids']:raise ValueError('document identity changed')
    heldout=set(previous_data_protocol['partitions']['dev'])|set(previous_data_protocol['partitions']['test'])
    forbidden={document_key(data.prompts[d],data.responses[d]) for d in heldout}
    allowed=[];duplicates=[]
    for doc in range(len(data)):
        if doc in heldout:continue
        if document_key(data.prompts[doc],data.responses[doc]) in forbidden:duplicates.append(doc)
        else:allowed.append(doc)
    caps=previous_data_protocol['response_budget']['response_caps']
    units=unseen_response_units(data.responses,allowed,caps,window_bytes)
    metadata={'unit_format':['document_id','response_byte_offset','target_byte_count'],
        'units_sha256':hashlib.sha256(units.astype('<i8',copy=False).tobytes()).hexdigest(),
        'units':len(units),'new_target_bytes':int(units[:,2].sum()),
        'previous_target_bytes':sum(map(int,caps.values())),
        'documents_with_new_targets':int(len(np.unique(units[:,0]))),
        'prompt_max':prompt_max,'response_window':window_bytes,'excluded_text_duplicate_ids':duplicates,
        'scope':'new target bytes only; previous prefixes may appear as causal context; test not evaluated'}
    return ResponseWindows(data,units,prompt_max),metadata
