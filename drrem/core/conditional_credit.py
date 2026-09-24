"""State-conditioned error translation, trained on exact consumer adjoints.

The low-rank bilinear term combines an error projection (apical signal) with
a projection of current states (content). It models a state-dependent local
Jacobian, rather than a scalar confidence or activation magnitude. This is a
synthetic-gradient model; its success must be judged on future blocks.
"""
import torch
from torch import nn

from drrem.core.synthetic_credit import SyntheticCreditReader


class ConditionalFeedback(nn.Module):
    def __init__(self,maps,rank=16):
        super().__init__();maps=torch.stack(maps)
        self.register_buffer('maps',maps)
        self.layers,self.width=maps.shape[:2];self.rank=rank
        self.error=nn.Parameter(torch.randn(self.layers,self.width,rank))
        self.content=nn.Parameter(torch.randn(self.layers,self.layers*self.width,rank)/(self.layers*self.width)**.5)
        self.values=nn.Parameter(torch.zeros(self.layers,rank,self.width))

    @staticmethod
    def context(states):
        if isinstance(states,(tuple,list)):states=torch.stack(states,-2)
        scale=states.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        return (states.float()/scale).flatten(-2)

    def forward(self,error,context,level):
        error=error.float();context=context.float()
        base=error@self.maps[level]
        correction=((error@self.error[level])*torch.tanh(context@self.content[level]))@self.values[level]
        return base+correction


class ConditionalCreditReader(SyntheticCreditReader):
    def __init__(self,model,moments,feedback,scales,cut=4,rate=3e-4):
        self.feedback=feedback.eval().requires_grad_(False)
        self.current_states=None
        super().__init__(model,moments,list(feedback.maps),scales,cut=cut,rate=rate)

    def read(self,x,loss_mask,active):
        original=self.model.after_hop;owned='after_hop' in vars(self.model)
        def capture(states,hop):
            states=original(states,hop)
            if hop==self.cut:self.current_states=tuple(s.detach() for s in states)
            return states
        self.model.after_hop=capture
        try:return super().read(x,loss_mask,active)
        finally:
            if owned:self.model.after_hop=original
            else:del self.model.after_hop
            self.current_states=None

    def local_gradients(self,output_error,inputs):
        with torch.no_grad(),torch.autocast(output_error.device.type,enabled=False):
            context=self.feedback.context(self.current_states)
            credits=[self.feedback(output_error,context,i)*self.scales[i]*self.model.step_scale
                     for i in range(self.model.cfg.layers)]
        gradients={}
        for level,(u,credit) in enumerate(zip(inputs,credits)):
            with torch.autocast(u.device.type,dtype=torch.bfloat16,enabled=u.is_cuda):
                proposal=self.model.neurons[level](u.detach())
            energy=(proposal.float()*credit.detach()).sum()
            named=list(self.model.neurons[level].named_parameters())
            grads=torch.autograd.grad(energy,[p for _,p in named])
            gradients.update({f'neurons.{level}.{n}':g.float() for (n,_),g in zip(named,grads)})
        return [gradients[n] for n in self.names]
