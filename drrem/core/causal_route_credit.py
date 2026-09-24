"""Exact same-position transport credit, delayed until its target is observed.

A naive backward over all observed errors is NOT causal evidence: attention
also sends later-position losses into earlier states. Here each replayed
temporal query is live, but keys/values come from the recorded first path,
outside the replay seed variables. This extracts the diagonal time blocks of
the Jacobian while retaining all dense spatial paths and nonlinearities.

The outer derivative is retained, including through recorded K/V and replay
seed values. This is a diagnostic/operator building block, not yet evidence
that such a second-order correction improves language modeling.

Exact Jacobian parity is tested in FP32. Under mixed-precision training the
replay uses FP32 around the recorded states; it is not claimed bit-identical
to the autograd arithmetic of the BF16 first solve.
"""
import math

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from drrem.core.causal_transport import rotate, CausalTransportMachine
from drrem.core.directed_flywheel import directed_evidence, DirectedFlywheelMachine
from drrem.core.semantic_flywheel import delay


def geometry(model, ids, valid):
    t=ids.shape[1]
    causal=torch.ones(t,t,device=ids.device,dtype=torch.bool).tril(-1)
    if model.cfg.window:causal=causal.triu(-model.cfg.window)
    mask=causal[None,None]
    mask=mask&valid[:,None,None,:]&valid[:,None,:,None]
    d=model.cfg.neurons//model.cfg.heads
    frequencies=10000.**(-torch.arange(0,d,2,device=ids.device,dtype=torch.float32)/d)
    phase=torch.arange(t,device=ids.device)[:,None]*frequencies
    return mask,phase.cos(),phase.sin()


def replay_hop(model, states, reference, valid, mask, cosine, sine):
    """Same forward at the reference point; only diagonal temporal derivatives."""
    normalized=[norm(s) for norm,s in zip(model.source_norm,states)]
    reference_norm=[norm(s) for norm,s in zip(model.source_norm,reference)]
    outputs=[]
    n=model.cfg.neurons; heads=model.cfg.heads
    for i,x in enumerate(states):
        sources=range(max(0,i-1),min(model.cfg.layers,i+2))
        field=sum(model.edge_gains[f'{i}_{j}']*model.edges[f'{i}_{j}'](normalized[j]) for j in sources)/math.sqrt(len(sources))
        temporal=model.temporal[i]
        q=F.linear(normalized[i],temporal.qkv.weight[:n])
        k,v=F.linear(reference_norm[i],temporal.qkv.weight[n:]).chunk(2,-1)
        b,t,_=x.shape
        q,k,v=(a.reshape(b,t,heads,n//heads).transpose(1,2) for a in (q,k,v))
        if model.cfg.rotary:
            q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
        # Math SDPA supports the outer derivative through this exact VJP.
        with sdpa_kernel(SDPBackend.MATH):
            read=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        field=field+temporal.out(read.transpose(1,2).reshape_as(x))
        proposal=field+model.neurons[i](model.field_norm[i](x+model.step_scale*field))
        outputs.append((x+model.step_scale*proposal)*valid[...,None])
    return tuple(outputs)


def route_credit(model, trajectory, first_logits, ids, valid, start_hop=3, horizons=1, create_graph=None):
    """Return B,T,H,L,N negative CE derivatives in their SOURCE coordinates.

Credit[t,h,l] refers to state[l] at position t-h-1 AFTER start_hop first-solve
hops. Every key/value in its dependency graph is at or before that forecast
origin, and the only later datum is the now-observed target x[t].

This is NOT credit at the final states and must not be labeled that way by a
consumer. If applied there, an explicit learned coordinate transfer is needed.
"""
    if model.cfg.history!='attention' or model.cfg.hop_rule!='residual':
        raise ValueError('causal residual attention transport required')
    implementation=lambda method:getattr(method,'__func__',method)
    if (implementation(model.transport_hop) is not CausalTransportMachine.transport_hop
            or implementation(model.after_hop) is not CausalTransportMachine.after_hop
            or implementation(model.neuron_response) is not CausalTransportMachine.neuron_response
            or implementation(getattr(model,'decode',None)) not in (None,DirectedFlywheelMachine.decode)):
        raise ValueError('route-credit replay supports the plain transport and final linear decoder only')
    if len(trajectory)!=model.cfg.hops+1 or not 0<=start_hop<=model.cfg.hops-model.cfg.layers:
        raise ValueError('all levels need a complete spatial path to the final decoder')
    outer_grad=torch.is_grad_enabled()
    create_graph=outer_grad if create_graph is None else create_graph
    if create_graph and not outer_grad:
        raise ValueError('cannot request an outer graph from a no-grad caller')
    with torch.enable_grad(), torch.autocast(ids.device.type,enabled=False):
        mask,cosine,sine=geometry(model,ids,valid)
        evidence,mature=directed_evidence(first_logits,trajectory[-1][-1],ids,valid,
            model.readout,model.final_norm.weight,horizons,'state_credit')
        seeds=tuple(s.float().clone().requires_grad_(True) for s in trajectory[start_hop])
        states=seeds
        for hop in range(start_hop,model.cfg.hops):
            reference=tuple(s.float() for s in trajectory[hop])
            states=replay_hop(model,states,reference,valid,mask,cosine,sine)
        credits=[]
        for h in range(horizons):
            lag=h+1
            g=evidence[:,:,h,:model.cfg.neurons]
            # Align the already delayed, matured derivative with its origin.
            g=torch.cat((g[:,lag:],torch.zeros_like(g[:,:lag])),1) if lag<g.shape[1] else torch.zeros_like(g)
            vjp=torch.autograd.grad(states[-1],seeds,grad_outputs=g,create_graph=create_graph,
                retain_graph=create_graph or h+1<horizons,allow_unused=False)
            credits.append(torch.stack([delay(v,lag) for v in vjp],2)*mature[:,:,h,None,None])
        result=torch.stack(credits,2)
    return result if outer_grad else result.detach()
