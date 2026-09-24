"""An end-to-end trained, causal quadratic energy for dense state agreement.

E(z|a) = .5 sum_i tau_i(a_i)*(z_i-a_i)^2
       + .5*kappa sum_(target,source) ||z_target-P_ts z_source||^2.

All seven dense operators remain distinct. Intra-level terms are included.
Each position has its own positive, learned precision and independent CG
coefficients. Content anchors are fixed during the solve; currents are the
explicit inter-level prediction discrepancies. Ordinary outer Adam learns
through the solver and final CE, not a separately fitted quality judge.

Strict convexity certifies this inner energy's unique minimum, NOT its
usefulness for language. Whether further minimization helps CE is a required
empirical test. Four additional CG iterations at the middle of transport
are explicit extra computation, followed by the real remaining native hops.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportMachine


class EnergyConsensusTransportMachine(CausalTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.precision_bias=nn.Parameter(torch.full((cfg.layers,cfg.neurons),math.log(math.expm1(.9))))
        self.precision_slope=nn.Parameter(torch.zeros(cfg.layers,cfg.neurons))
        self.energy_steps=4;self.coupling=.1

    def energy_context(self,states):
        scales=tuple((s.float().square().mean(-1,keepdim=True)+1e-5).sqrt() for s in states)
        anchors=tuple(s.float()/r for s,r in zip(states,scales))
        precision=tuple(F.softplus(self.precision_bias[i]+self.precision_slope[i]*a.tanh())+.1 for i,a in enumerate(anchors))
        operators={name:edge.weight.float()/(edge.weight.float().square().sum()/self.cfg.neurons+1e-8).sqrt()
                   for name,edge in self.edges.items()}
        return anchors,scales,precision,operators

    def energy(self,z,anchors,precision,operators):
        value=sum((p*(x-a).square()).sum(-1) for x,a,p in zip(z,anchors,precision))
        for name,weight in operators.items():
            target,source=map(int,name.split('_'))
            error=z[target]-F.linear(z[source],weight)
            value=value+self.coupling*error.square().sum(-1)
        return .5*value/self.cfg.neurons

    def hessian_action(self,z,precision,operators):
        result=[p*x for p,x in zip(precision,z)]
        for name,weight in operators.items():
            target,source=map(int,name.split('_'))
            error=z[target]-F.linear(z[source],weight)
            result[target]=result[target]+self.coupling*error
            result[source]=result[source]-self.coupling*F.linear(error,weight.T)
        return tuple(result)

    def diagonal(self,precision,operators):
        result=list(precision);eye=torch.eye(self.cfg.neurons,device=precision[0].device)
        for name,weight in operators.items():
            target,source=map(int,name.split('_'))
            if target==source:result[source]=result[source]+self.coupling*(eye-weight).square().sum(0)
            else:
                result[target]=result[target]+self.coupling
                result[source]=result[source]+self.coupling*weight.square().sum(0)
        return tuple(result)

    def solve_energy(self,states,return_trace=False):
        with torch.autocast(states[0].device.type,enabled=False):
            anchors,scales,precision,operators=self.energy_context(states)
            z=anchors;rhs=tuple(p*a for p,a in zip(precision,anchors))
            hz=self.hessian_action(z,precision,operators);residual=tuple(b-h for b,h in zip(rhs,hz))
            diagonal=self.diagonal(precision,operators)
            pre=tuple(r/d for r,d in zip(residual,diagonal));direction=pre
            dot=lambda x,y:sum((a*b).sum(-1,keepdim=True) for a,b in zip(x,y))
            rz=dot(residual,pre);tolerance=rz.detach()*1e-10+1e-20
            trace=[z] if return_trace else None
            for _ in range(self.energy_steps):
                hd=self.hessian_action(direction,precision,operators)
                active=rz>tolerance
                alpha=torch.where(active,rz/dot(direction,hd).clamp_min(1e-20),0.)
                z=tuple(x+alpha*d for x,d in zip(z,direction))
                residual=tuple(r-alpha*h for r,h in zip(residual,hd))
                pre=tuple(r/d for r,d in zip(residual,diagonal));new_rz=dot(residual,pre)
                beta=torch.where(active&(new_rz>tolerance),new_rz/rz.clamp_min(1e-20),0.)
                direction=tuple(p+beta*d for p,d in zip(pre,direction));rz=new_rz
                if return_trace:trace.append(z)
            result=tuple(x*s for x,s in zip(z,scales))
        if return_trace:return result,trace,(anchors,precision,operators)
        return result

    def after_hop(self,states,hop):
        if self.energy_steps and hop==self.cfg.hops//2:return self.solve_energy(states)
        return states
