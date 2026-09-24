"""Local Forward-Forward diagnostic on observed document blocks.

Positive data are real ordered bytes; negative data reverse short byte groups
within the same row, preserving the exact byte histogram and padding. Each
level's selected MLP occurrence sees detached positive/negative inputs and
learns high/low squared-activation goodness using ordinary Adam. No global
error derivative, extra decoder or teacher is supplied to this rule.

The ordinary final decoder scores each block BEFORE any FF update. Therefore
the experiment asks whether order-discrimination learning transfers to later
language predictions. A negative result tests this corruption/energy choice,
not all possible FF designs.
"""
import torch
from torch.nn import functional as F

from drrem.core.plastic_reader import PlasticReader, horizon_nats
from drrem.core.causal_transport import response_objective
from drrem.diagnostics.consumer_replay import geometry


def reverse_groups(ids, valid, width=16):
    out = ids.clone()
    for row in range(len(ids)):
        indices = (valid[row] & (ids[row] < 256)).nonzero().flatten()
        for start in range(0, len(indices), width):
            group = indices[start:start+width]
            out[row, group] = ids[row, group.flip(0)]
    return out


def reorder_chunks(ids, valid, width=32):
    """Keep bytes inside chunks ordered, reverse only the chunk sequence."""
    out=ids.clone()
    for row in range(len(ids)):
        indices=(valid[row] & (ids[row]<256)).nonzero().flatten()
        chunks=list(indices.split(width))
        if chunks:out[row,indices]=ids[row,torch.cat(list(reversed(chunks)))]
    return out


class ForwardForwardReader(PlasticReader):
    def __init__(self, model, moments, cut=4, rate=3e-4, threshold=2.,
                 corruption='reverse_bytes', decoder_rate=0., loss_rule='threshold'):
        if not 1 <= cut <= model.cfg.hops:
            raise ValueError('FF must name a real MLP occurrence')
        if corruption not in ('reverse_bytes','chunk_order'):raise ValueError('unknown FF corruption')
        if loss_rule not in ('threshold','pairwise'):raise ValueError('unknown FF loss')
        self.loss_rule=loss_rule
        self.cut = cut; self.threshold = threshold; self.goodness_scales = None
        self.corruption=corruption;self.ff_rate=rate;self.decoder_rate=decoder_rate
        self.all_goodness_trace=[]
        self.goodness_trace = []
        self.decoder_names={'readout','final_norm.weight','ridge_raw','plastic_gain','plastic_address.weight'}
        rates={f'level{i}':rate for i in range(model.cfg.layers)}
        rates['readout']=decoder_rate
        super().__init__(model, moments, rate=rates, matrix_rule='torch_adam',
                         scope=lambda n:n.startswith('neurons.') or (decoder_rate>0 and n in self.decoder_names))

    def begin_document(self):
        super().begin_document()
        self.goodness_scales = None; self.goodness_trace = []

    def local_loss(self,gp,gn):
        if self.loss_rule=='pairwise':return F.softplus(gn-gp).mean()
        return F.softplus(self.threshold-gp).mean()+F.softplus(gn-self.threshold).mean()

    @torch.no_grad()
    def capture(self, ids, valid, complete):
        calls = [0]*self.model.cfg.layers; inputs = [None]*self.model.cfg.layers; hooks=[]
        for level, neurons in enumerate(self.model.neurons):
            def keep(_module, args, i=level):
                calls[i] += 1
                if calls[i] == self.cut:inputs[i] = args[0].detach()
            hooks.append(neurons.register_forward_pre_hook(keep))
        try:
            with torch.autocast(ids.device.type, dtype=torch.bfloat16, enabled=ids.is_cuda):
                if complete:
                    states=self.model.forward_states(ids,valid)
                    self._final_state=states[-1].detach()
                    logits = self.model.decode(states[-1],ids,valid)
                else:
                    x = self.model.encode_input(ids, valid)
                    states = (x,) + tuple(torch.zeros_like(x) for _ in range(self.model.cfg.layers-1))
                    geo = geometry(self.model, ids, valid)
                    for hop in range(self.cut):
                        states = self.model.transport_hop(states, valid, *geo)
                        states = self.model.after_hop(states, hop+1)
                    logits = None
        finally:
            for hook in hooks:hook.remove()
        return logits, inputs

    def read(self, x, loss_mask, active):
        ids=x[:,:-1]; valid=active[:,:-1]
        logits, positive = self.capture(ids, valid, True)
        nats, counts = horizon_nats(logits, x, loss_mask, active)
        if self.rate0 <= 0:return logits, nats, counts
        negative=None
        if self.ff_rate>0:
            corrupt=reverse_groups if self.corruption=='reverse_bytes' else reorder_chunks
            _, negative = self.capture(corrupt(ids, valid), valid, False)
        selected=loss_mask[:,:-1]&valid
        gradient={}; statistics=[]
        if self.goodness_scales is None:self.goodness_scales=[None]*self.model.cfg.layers
        for level, (p,n) in enumerate(zip(positive,negative or [])):
            with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                zp=self.model.neurons[level](p.detach());zn=self.model.neurons[level](n.detach())
            gp=zp.float().square().mean(-1)[selected];gn=zn.float().square().mean(-1)[selected]
            if self.goodness_scales[level] is None:
                # Fixed scale established after the first block is scored;
                # no gradients through it and no updating batch normalization.
                self.goodness_scales[level]=.5*(gp.detach().mean()+gn.detach().mean()).clamp_min(1e-8)
            scale=self.goodness_scales[level]
            gp=gp/scale;gn=gn/scale
            loss=self.local_loss(gp,gn)
            named=list(self.model.neurons[level].named_parameters())
            grads=torch.autograd.grad(loss,[q for _,q in named])
            gradient.update({f'neurons.{level}.{name}':g.float() for (name,_),g in zip(named,grads)})
            statistics.append(dict(loss=float(loss.detach()),positive=float(gp.detach().mean()),negative=float(gn.detach().mean())))
        if self.decoder_rate>0:
            named=[(n,p) for n,p in zip(self.names,self.params) if n in self.decoder_names]
            with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                decoded=self.model.decode(self._final_state,ids,valid)
                loss,_,_=response_objective(decoded,x,loss_mask[:,:-1],valid)
            grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
            gradient.update({n:g.float() if g is not None else torch.zeros_like(p) for (n,p),g in zip(named,grads)})
        self.step([gradient[n] if n in gradient else torch.zeros_like(p) for n,p in zip(self.names,self.params)]);self.updates+=1
        # Measure the very same local objective with frozen inputs after its
        # step. This tests descent of the FF objective, not a changed batch.
        with torch.no_grad():
            for level,(p,n) in enumerate(zip(positive,negative or [])):
                with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                    zp=self.model.neurons[level](p);zn=self.model.neurons[level](n)
                gp=zp.float().square().mean(-1)[selected]/self.goodness_scales[level]
                gn=zn.float().square().mean(-1)[selected]/self.goodness_scales[level]
                after=self.local_loss(gp,gn)
                statistics[level]['after_loss']=float(after)
                statistics[level]['after_positive']=float(gp.mean());statistics[level]['after_negative']=float(gn.mean())
        self.goodness_trace.append(statistics);self.all_goodness_trace.append(statistics)
        return logits,nats,counts
