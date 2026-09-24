"""A diagnostic control: exact error, but credit only one shared MLP use.

This deliberately pays the full consumer backward. Comparing it with the
fitted feedback map distinguishes map error from omitted hop contributions.
It is not the proposed cheap reading algorithm.
"""
import torch

from drrem.core.causal_transport import response_objective
from drrem.core.plastic_reader import PlasticReader, horizon_nats


class ExactOccurrenceReader(PlasticReader):
    def __init__(self, model, moments, cut=4, rate=3e-4):
        if not 1<=cut<=model.cfg.hops:raise ValueError('invalid occurrence')
        self.cut=cut
        super().__init__(model,moments,rate=rate,matrix_rule='torch_adam',scope=lambda n:n.startswith('neurons.'))

    def read(self,x,loss_mask,active):
        inputs=[None]*self.model.cfg.layers;outputs=list(inputs);counts=[0]*len(inputs);hooks=[]
        for level,neurons in enumerate(self.model.neurons):
            def capture(_module,args,output,i=level):
                counts[i]+=1
                if counts[i]==self.cut:inputs[i]=args[0].detach();outputs[i]=output
            hooks.append(neurons.register_forward_hook(capture))
        try:
            with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                logits=self.model(x[:,:-1],active[:,:-1])
                energy,_,_=response_objective(logits,x,loss_mask[:,:-1],active[:,:-1])
        finally:
            for hook in hooks:hook.remove()
        nats,counts=horizon_nats(logits.detach(),x,loss_mask,active)
        errors=torch.autograd.grad(energy,outputs,allow_unused=True)
        gradients={}
        for level,(u,g) in enumerate(zip(inputs,errors)):
            named=list(self.model.neurons[level].named_parameters())
            if g is None:
                gradients.update({f'neurons.{level}.{n}':torch.zeros_like(p) for n,p in named});continue
            with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
                proposal=self.model.neurons[level](u)
            loss=(proposal.float()*g.detach().float()).sum()
            grads=torch.autograd.grad(loss,[p for _,p in named])
            gradients.update({f'neurons.{level}.{n}':v.float() for (n,_),v in zip(named,grads)})
        self.step([gradients[n] for n in self.names]);self.updates+=1
        return logits.detach(),nats,counts
