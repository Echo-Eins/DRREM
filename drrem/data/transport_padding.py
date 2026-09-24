"""Execution-only padding for transport batches; all added positions are inert."""
import numpy as np
import torch.nn.functional as F
from drrem.data.openorca import Batch


def pad_transport_batch(batch,length,rows):
    extra_time=length-batch.x.shape[1];extra_rows=rows-batch.x.shape[0]
    if min(extra_time,extra_rows)<0:raise ValueError('padding may not truncate data')
    if not extra_time and not extra_rows:return batch
    padding=(0,extra_time,0,extra_rows)
    ids=np.concatenate((batch.doc_ids,np.full(extra_rows,-1,dtype=batch.doc_ids.dtype)))
    return Batch(F.pad(batch.x,padding,value=0),F.pad(batch.loss_mask,padding,value=False),
                 F.pad(batch.active,padding,value=False),batch.P,ids)
