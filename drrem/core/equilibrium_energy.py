"""Train the energy minimum itself, rather than a fixed solver transient.

The positive state-dependent diagonal is split from a shared dense SPD
matrix. A Cholesky preconditioner makes the per-position converged solve
practical. Backward solves the adjoint system and applies the exact implicit
derivative of Hx=b; it does not retain or differentiate a truncated CG trace.
Forward and adjoint residuals are checked. No future byte is a solve input.
"""
import torch
from torch.nn import functional as F
from drrem.core.energy_consensus import EnergyConsensusTransportMachine


def converged_solve(base,delta,rhs,lower,rtol=None,max_steps=32):
    rtol=(1e-11 if base.dtype==torch.float64 else 2e-5) if rtol is None else rtol
    # Adjoint rows can be tiny. Normalize each independent RHS so an absolute
    # floating-point floor cannot silently accept a poor relative solution.
    scale=rhs.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(rhs.dtype).tiny)
    rhs=rhs/scale
    action=lambda x:F.linear(x,base)+delta*x
    precondition=lambda x:torch.cholesky_solve(x.T.contiguous(),lower).T
    x=precondition(rhs);r=rhs-action(x)
    threshold=rhs.square().sum(-1,keepdim=True)*rtol**2
    active=r.square().sum(-1,keepdim=True)>threshold
    z=precondition(r);direction=torch.where(active,z,0.);rz=(r*z).sum(-1,keepdim=True)
    counts=torch.zeros_like(active,dtype=torch.int32)
    for _ in range(max_steps):
        if not bool(active.any()):break
        hd=action(direction);denominator=(direction*hd).sum(-1,keepdim=True)
        alpha=torch.where(active,rz/denominator.clamp_min(1e-30),0.)
        x=x+alpha*direction;counts=counts+active.to(torch.int32)
        r=rhs-action(x);next_active=active&(r.square().sum(-1,keepdim=True)>threshold)
        z=precondition(r);next_rz=(r*z).sum(-1,keepdim=True)
        beta=torch.where(next_active,next_rz/rz.clamp_min(1e-30),0.)
        direction=torch.where(next_active,z+beta*direction,0.);rz=next_rz;active=next_active
    relative=(rhs-action(x)).square().sum(-1).sqrt()/rhs.square().sum(-1).sqrt().clamp_min(1e-20)
    if not bool(torch.isfinite(relative).all()) or bool(active.any()):
        raise RuntimeError(f'equilibrium solve failed residual contract: max={float(relative.max()):.6g}, rtol={rtol}')
    return x*scale,relative,counts.squeeze(-1)


class EquilibriumSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,base,delta,rhs):
        lower=torch.linalg.cholesky(base)
        x,residual,counts=converged_solve(base,delta,rhs,lower)
        ctx.save_for_backward(base,delta,lower,x)
        ctx.mark_non_differentiable(residual,counts)
        return x,residual,counts

    @staticmethod
    def backward(ctx,gradient,_residual,_counts):
        base,delta,lower,x=ctx.saved_tensors
        adjoint,_,_=converged_solve(base,delta,gradient.contiguous(),lower)
        # The upstream matrix is symmetric by construction.
        matrix_gradient=-adjoint.T@x
        matrix_gradient=(matrix_gradient+matrix_gradient.T)*.5
        return matrix_gradient,-adjoint*x,adjoint


class EquilibriumEnergyTransportMachine(EnergyConsensusTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.last_equilibrium=None
        self._equilibrium_cache=None

    def energy_operator_parameters(self):
        return [edge.weight for edge in self.edges.values()]

    def shared_hessian(self,operators):
        n,l=self.cfg.neurons,self.cfg.layers
        eye=torch.eye(n,device=self.precision_bias.device);zero=torch.zeros_like(eye)
        precision0=F.softplus(self.precision_bias)+.1
        blocks=[[torch.diag(precision0[i]) if i==j else zero for j in range(l)] for i in range(l)]
        for name,weight in operators.items():
            target,source=map(int,name.split('_'));c=self.coupling
            if target==source:
                difference=eye-weight
                blocks[source][source]=blocks[source][source]+c*(difference.T@difference)
            else:
                blocks[target][target]=blocks[target][target]+c*eye
                blocks[source][source]=blocks[source][source]+c*(weight.T@weight)
                blocks[target][source]=blocks[target][source]-c*weight
                blocks[source][target]=blocks[source][target]-c*weight.T
        return torch.cat([torch.cat(row,1) for row in blocks],0),precision0

    def solve_energy(self,states,return_trace=False):
        if return_trace:raise ValueError('an implicit equilibrium has no truncated trajectory; use residual diagnostics')
        with torch.autocast(states[0].device.type,enabled=False):
            anchors,scales,precision,operators=self.energy_context(states)
            inference=not self.training and not torch.is_grad_enabled()
            key=(tuple(p._version for p in [self.precision_bias]+self.energy_operator_parameters()),
                 self.precision_bias.device,self.precision_bias.dtype,self.coupling)
            if inference and self._equilibrium_cache is not None and self._equilibrium_cache[0]==key:
                _,base,p0,lower=self._equilibrium_cache
            else:
                base,p0=self.shared_hessian(operators)
                if inference:
                    lower=torch.linalg.cholesky(base)
                    self._equilibrium_cache=(key,base,p0,lower)
            rhs=torch.cat([p*a for p,a in zip(precision,anchors)],-1)
            delta=torch.cat([p-p0[i] for i,p in enumerate(precision)],-1)
            shape=rhs.shape
            flat_delta,flat_rhs=delta.reshape(-1,shape[-1]),rhs.reshape(-1,shape[-1])
            if inference:solution,residual,counts=converged_solve(base,flat_delta,flat_rhs,lower)
            else:solution,residual,counts=EquilibriumSolve.apply(base,flat_delta,flat_rhs)
            self.last_equilibrium=dict(max_relative_residual=residual.max().detach(),mean_refinement_steps=counts.float().mean().detach(),
                                       max_refinement_steps=counts.max().detach())
            unit_states=solution.reshape(shape).split(self.cfg.neurons,-1)
            return tuple(x*s for x,s in zip(unit_states,scales))
