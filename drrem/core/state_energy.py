"""A causal local quality surrogate trained against the frozen downstream consumer.

Energy is NOT activation magnitude or confidence. It predicts changes in final
next-byte cross entropy under controlled state interventions. Context is held
fixed while optimizing a candidate, and no target byte enters this function.
Its minima require empirical downstream validation; low energy is not a proof.
"""
import torch
from torch import nn


class StateQualityEnergy(nn.Module):
    def __init__(self,width,layers=3,hidden=128):
        super().__init__();self.width=width;self.layers=layers
        self.networks=nn.ModuleList([nn.Sequential(nn.Linear((layers+1)*width+layers,hidden),nn.SiLU(),
                        nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1)) for _ in range(layers)])

    def context_features(self,states):
        rms=[(x.float().square().mean(-1,keepdim=True)+1.).sqrt() for x in states]
        features=torch.cat([x.float()/r for x,r in zip(states,rms)]+[r.log() for r in rms],-1)
        return features,rms

    def forward(self,candidate,states,level):
        context,rms=self.context_features(states)
        inp=torch.cat((candidate.float()/rms[level],context),-1)
        return self.networks[level](inp).squeeze(-1)


def minimize_energy(energy,states,level,steps=8,lr=.03,radius=.3):
    """Ordinary Adam on dimensionless state displacement, bounded per position.

    Each row/position has an independent derivative and trust ball. There is
    no shared batch/prefix stopping or normalization that could leak a future
    suffix into an earlier prediction. Judge weights and context stay fixed.
    """
    anchor=tuple(s.detach() for s in states);base=anchor[level]
    rms=(base.float().square().mean(-1,keepdim=True)+1.).sqrt()
    displacement=nn.Parameter(torch.zeros_like(base,dtype=torch.float32))
    optimizer=torch.optim.Adam([displacement],lr=lr)
    trajectory=[]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        candidate=base+rms*displacement
        # grad rather than backward: judge parameters never accumulate updates.
        gradient=torch.autograd.grad(energy(candidate,anchor,level).sum(),displacement)[0]
        displacement.grad=gradient
        previous=displacement.detach().clone()
        with torch.no_grad():old_energy=energy(base+rms*previous,anchor,level)
        optimizer.step()
        with torch.no_grad():
            length=displacement.square().mean(-1,keepdim=True).sqrt()
            displacement.mul_((radius/length.clamp_min(1e-12)).clamp_max(1.))
            proposal=displacement.detach().clone();chosen=previous.clone();accepted=torch.zeros_like(old_energy,dtype=torch.bool)
            for scale in [1.,.5,.25,.125]:
                trial=previous+scale*(proposal-previous)
                score=energy(base+rms*trial,anchor,level)
                take=(score<=old_energy)&torch.isfinite(score)&~accepted
                chosen=torch.where(take[...,None],trial,chosen);accepted|=take
            displacement.copy_(chosen)
            trajectory.append((base+rms*displacement).detach().clone())
    return trajectory
