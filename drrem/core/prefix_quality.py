"""A target-free, anchored state-quality energy with an explicit consumer.

The fitted kernel predicts derivatives of the real remaining model. Context
consists of causal states at two consecutive hops and is FIXED during state
minimization. The quadratic trust term makes the local minimum well-defined;
it does not certify a minimum of the real language loss. No future label,
decoder error, batch statistic or document memory enters prediction.
"""
import torch
from torch import nn


def prefix_features(point, next_point):
    scale=point.float().square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    velocity=next_point.float()-point.float()
    return torch.cat((point.float()/scale,velocity/scale,scale.log()),-1).flatten(-2)


class KernelPrefixEnergy(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.kinds=tuple(state['kinds'])
        for name in ('mean','std','train_features','weights','bias'):
            self.register_buffer(name,state[name].clone())

    def predict(self,point,next_point):
        x=(prefix_features(point,next_point)-self.mean)/self.std
        support=self.train_features
        dot=x@support.T/x.shape[-1]
        distance=(x.square().mean(-1)[...,None]+support.square().mean(-1)-2*dot).clamp_min(0)
        out=[]
        for i,kind in enumerate(self.kinds):
            if kind=='linear': k=dot
            elif kind.startswith('rbf:'): k=(-distance/float(kind.split(':')[1])).exp()
            else: raise ValueError('this fitted judge supports linear/RBF kernels')
            out.append(k@self.weights[i]+self.bias[i])
        return torch.stack(out,-2)

    def direction(self,point,next_point,level):
        p=point[...,level,:].float()
        g=self.predict(point,next_point)[...,level,:]
        g=g-p*(g*p).sum(-1,keepdim=True)/p.square().sum(-1,keepdim=True).clamp_min(1e-30)
        return g/g.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)

    def forward(self,candidate,point,next_point,level,radius=.03):
        if radius<=0:raise ValueError('positive trust radius required')
        # The anchor is a separate conditioning input, not the variable being
        # optimized; moving it while descending would change the problem.
        point,next_point=point.detach(),next_point.detach()
        base=point[...,level,:].float()
        scale=base.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
        delta=(candidate.float()-base)/scale
        g=self.direction(point,next_point,level)
        return (g*delta).mean(-1)+delta.square().mean(-1)/(2*radius)

    def minimum(self,point,next_point,level,radius=.03):
        if radius<=0:raise ValueError('positive trust radius required')
        base=point[...,level,:].float()
        scale=base.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
        return base-radius*scale*self.direction(point,next_point,level)
