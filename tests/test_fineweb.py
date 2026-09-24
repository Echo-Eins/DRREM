import numpy as np
import torch
from drrem.data.fineweb import FineWebBytes,BOUNDARY,window_batch
from drrem.core.causal_transport import response_objective


class TinyCorpus:
    splits={'train':np.arange(3)}
    documents=[np.frombuffer(x,dtype=np.uint8) for x in [b'abcdefghij','two'.encode(),b'XYZ123']]
    def document(self,i):return self.documents[i]
    plan=FineWebBytes.plan


def test_all_bytes_exactly_once_and_no_false_eos_at_budget_cut():
    c=TinyCorpus();p=c.plan(budget=16,block=3,context=4)
    assert p['raw_bytes']==16 and p['boundary_targets']==2
    collected={}
    for i,u in enumerate(p['units']):
        b=window_batch(c,p,[i]);target=b.x[:,1:][b.loss_mask[:,:-1]].tolist()
        collected.setdefault(u[0],[]).extend(target)
    assert collected=={0:list(b'abcdefghij')+[BOUNDARY],1:list(b'two')+[BOUNDARY],2:list(b'XYZ')}


def test_context_is_prior_document_bytes_and_padding_is_inactive():
    c=TinyCorpus();p=c.plan(budget=19,block=3,context=4)
    for i,(doc,start,count,cap,complete) in enumerate(p['units']):
        b=window_batch(c,p,[i]);active=b.active[0]
        observed=b.x[0,:b.P][active[:b.P]].tolist()
        expected=([BOUNDARY]+c.document(doc)[:start].tolist())[-p['context']:]
        assert observed==expected
        assert b.loss_mask.sum()==count


def test_horizons_do_not_supervise_padding_or_next_document():
    c=TinyCorpus();p=c.plan(budget=16,block=3,context=4)
    # Last complete-document unit contains a byte and EOS, another has a
    # budget-cut tail. The objective may not invent later horizon targets.
    b=window_batch(c,p,[3,5]);t=b.x.shape[1]-1
    logits=torch.zeros(2,t,8,257,requires_grad=True)
    loss,_,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
    assert counts.tolist()==[3,1,0,0,0,0,0,0]
    loss.backward();assert torch.isfinite(logits.grad).all()
