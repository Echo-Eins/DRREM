"""Event-driven S/A RREM with all-pairs STDP and supervised teaching events.

The inference path is an adaptive leaky integrate-and-fire network. The ONLY
source of recurrent weight updates is the per-synapse STDP eligibility below.
The shared dictionary uses the exact softmax delta rule and the STDP signal
of tied input synapses. Optimizer: unmodified torch Adam. Supervised mode uses
a ReSuMe-inspired teacher-minus-actual pair update, not an asserted exact CE
gradient. No surrogate spike derivative, equilibrium relaxation or BPTT.
"""
from dataclasses import asdict,dataclass,field
import copy
import math

import torch
import torch.nn.functional as F

from drrem.core.event_stdp import PairSTDP,STDPConfig
from drrem.rrem_repaired import doc_end,targets,project


@dataclass
class SpikeConfig:
    N:int=256
    L:int=2
    hops:int=8
    horizons:int=8
    delays:tuple=(1,2,8,32)  # queue bins; arrival latency is d-1/2 microticks
    tau_membrane:float=4.
    tau_adaptation:float=32.
    tau_readout:float=8.
    adaptation:float=2.
    refractory:int=1
    threshold:float=1.
    tonic:float=1.5
    input_gain:float=8.
    recurrent_gain:float=.4
    readout_gain:float=4.
    target_rate:float=.15
    homeostasis:float=.002
    core_lr:float=.0001
    head_lr:float=.003
    input_credit:float=.001  # train-only scale calibration of STDP versus CE on tied E
    tie_input:bool=True
    freeze_core:bool=False
    freeze_head:bool=False
    learn_input:bool=True
    rule:str='stdp'  # 'reverse' reverses update sign, not the physical event clock
    learning:str='resume'  # teacher/actual spike-train STDP difference
    teacher_gain:float=1.
    seed:int=20260923
    device:str='cuda' if torch.cuda.is_available() else 'cpu'
    dtype:str='float32'
    stdp:STDPConfig=field(default_factory=STDPConfig)

    @property
    def D(self):return self.N*self.L

    def validate(self):
        self.stdp.validate()
        if min(self.N,self.L,self.hops,self.horizons)<1:raise ValueError('sizes must be positive')
        if not self.delays or any(int(d)!=d or d<1 for d in self.delays):raise ValueError('positive integer axonal delays required')
        if min(self.tau_membrane,self.tau_adaptation,self.tau_readout,self.threshold)<=0:raise ValueError('invalid neuron constants')
        if self.rule not in ('stdp','reverse','no_eligibility'):raise ValueError('unknown STDP ablation')
        if self.learning not in ('modulated','resume'):raise ValueError('unknown supervised STDP rule')
        if self.teacher_gain<=0:raise ValueError('positive teacher gain required')
        if self.refractory<0 or not 0<self.target_rate<1:raise ValueError('invalid excitability')


@dataclass
class SpikeState:
    u:torch.Tensor
    adaptation:torch.Tensor
    refractory:torch.Tensor
    history:torch.Tensor
    rate:torch.Tensor
    events:torch.Tensor
    ticks:torch.Tensor
    stdp:PairSTDP|None=None
    input_stdp:PairSTDP|None=None
    teacher:object=None


class SpikingRREM:
    @torch.no_grad()
    def __init__(self,cfg:SpikeConfig):
        cfg.validate();self.cfg=cfg;self.dev=torch.device(cfg.device);self.dtype=getattr(torch,cfg.dtype)
        rng=torch.Generator().manual_seed(cfg.seed)
        rand=lambda *shape:torch.randn(*shape,generator=rng,dtype=self.dtype).to(self.dev)
        D,N,L,M=cfg.D,cfg.N,cfg.L,len(cfg.delays)
        level=torch.arange(D,device=self.dev)//N
        self.mask=((level[:,None]-level[None,:]).abs()<=1).to(self.dtype)
        self.mask.fill_diagonal_(0)  # no autapses, including delayed ones
        self.S=project(rand(M,D,D)*cfg.recurrent_gain/math.sqrt(N*M),1,self.mask)
        self.A=project(rand(M,D,D)*cfg.recurrent_gain/math.sqrt(N*M),-1,self.mask)
        self.E=rand(cfg.horizons,256,N)/math.sqrt(N)
        self.E_bias=torch.zeros(cfg.horizons,256,device=self.dev,dtype=self.dtype)
        self.E_in=rand(256,N)/math.sqrt(N)
        self.theta=torch.full((D,),cfg.threshold,device=self.dev,dtype=self.dtype)
        self.tonic=cfg.tonic+.15*rand(D)
        # Fixed heterogeneity avoids an artificial synchronous population.
        tau=cfg.tau_membrane*torch.exp(.35*rand(D))
        self.leak=torch.exp(-1/tau)
        self.initial_u=torch.rand(D,generator=rng,dtype=self.dtype).to(self.dev)*cfg.threshold
        self.delay_index=torch.tensor(cfg.delays,device=self.dev)-1
        self.params={n:torch.nn.Parameter(getattr(self,n)) for n in ('S','A','E','E_bias','E_in')}
        self.groups=[]
        if not cfg.freeze_core:self.groups.append({'params':[self.params['S'],self.params['A']],'lr':cfg.core_lr})
        if not cfg.freeze_head:self.groups.append({'params':[self.params['E'],self.params['E_bias']],'lr':cfg.head_lr})
        if not cfg.tie_input and cfg.learn_input:self.groups.append({'params':[self.params['E_in']],'lr':cfg.core_lr})
        self.optimizer=torch.optim.Adam(self.groups,weight_decay=0.,foreach=False) if self.groups else None
        self.updates=0;self.seen_targets=0

    def input_weights(self):
        e=self.E[0] if self.cfg.tie_input else self.E_in
        return self.cfg.input_gain*e/e.norm(dim=-1,keepdim=True).clamp_min(1e-8)

    def init_state(self,batch,learning=False):
        c=self.cfg;D=c.D
        zeros=lambda *shape:torch.zeros(*shape,device=self.dev,dtype=self.dtype)
        pair=PairSTDP(batch,len(c.delays),D,D,cfg=c.stdp,device=self.dev,dtype=self.dtype) if learning and not c.freeze_core else None
        inp=PairSTDP(batch,1,c.N,256,cfg=c.stdp,device=self.dev,dtype=self.dtype) if learning and c.learn_input else None
        state=SpikeState(self.initial_u[None].expand(batch,-1).clone(),zeros(batch,D),
            torch.zeros(batch,D,device=self.dev,dtype=torch.long),zeros(batch,max(c.delays),D),
            zeros(batch,D),zeros(batch,D),zeros(batch),pair,inp)
        if learning and c.learning=='resume' and (pair is not None or inp is not None):
            # Desired postsynaptic events live in a separate teacher compartment.
            # Presynaptic arrivals ALWAYS come from the free network.
            state.teacher={'u':state.u.clone(),'adaptation':state.adaptation.clone(),
                'refractory':state.refractory.clone()}
        return state

    @torch.no_grad()
    def tick(self,state,byte,active,*,weights=None,input_weights=None,record=False,teacher_signal=None):
        c=self.cfg
        W=self.S+self.A if weights is None else weights
        table=self.input_weights() if input_weights is None else input_weights
        traces=[];byte_events=torch.zeros_like(state.events)
        for hop in range(c.hops):
            arrivals=state.history[:,self.delay_index]
            field=torch.einsum('bmi,mji->bj',arrivals,W)
            input_spikes=torch.zeros(len(byte),256,device=self.dev,dtype=self.dtype)
            if hop==0:
                input_spikes.scatter_(1,byte[:,None],1.)
                field[:,:c.N]+=table[byte]
            adaptation=math.exp(-1/c.tau_adaptation)*state.adaptation
            threshold=self.theta+c.adaptation*adaptation
            u=self.leak*state.u+(1-self.leak)*self.tonic+field
            spikes=((u>=threshold)&(state.refractory==0)&active[:,None]).to(self.dtype)
            u=u-threshold*spikes
            ref=torch.where(spikes.bool(),c.refractory,(state.refractory-1).clamp_min(0))
            adaptation+=(1-math.exp(-1/c.tau_adaptation))*spikes
            decay=math.exp(-1/c.tau_readout)
            rate=decay*state.rate+(1-decay)*spikes
            v=active[:,None]
            state.u.copy_(torch.where(v,u,state.u))
            state.adaptation.copy_(torch.where(v,adaptation,state.adaptation))
            state.refractory.copy_(torch.where(v,ref,state.refractory))
            state.rate.copy_(torch.where(v,rate,state.rate))
            plastic_post=spikes
            if state.teacher is not None:
                target_state=state.teacher
                ad=math.exp(-1/c.tau_adaptation)*target_state['adaptation']
                th=self.theta+c.adaptation*ad
                feedback=teacher_signal(self.features(state)) if teacher_signal else 0.
                tu=self.leak*target_state['u']+(1-self.leak)*self.tonic+field+c.teacher_gain*feedback
                ts=((tu>=th)&(target_state['refractory']==0)&active[:,None]).to(self.dtype)
                tu-=th*ts
                tr=torch.where(ts.bool(),c.refractory,(target_state['refractory']-1).clamp_min(0))
                ad+=(1-math.exp(-1/c.tau_adaptation))*ts
                for key,val in (('u',tu),('adaptation',ad),('refractory',tr)):
                    target_state[key].copy_(torch.where(v,val,target_state[key]))
                # Pair STDP is linear in the post train. Track the FULL edge
                # difference directly: e(pre,target-actual)=e(target)-e(actual).
                # Both kernel lobes and every past pair remain present. This
                # saves one B*M*D*D tensor and avoids subtracting large traces.
                plastic_post=ts-spikes
            if state.stdp is not None:
                state.stdp.arrive_and_fire(arrivals,plastic_post,active)
                if c.rule=='no_eligibility':state.stdp.eligibility.zero_()
            if state.input_stdp is not None:
                state.input_stdp.arrive_and_fire(input_spikes[:,None],plastic_post[:,:c.N],active)
                if c.rule=='no_eligibility':state.input_stdp.eligibility.zero_()
            history=torch.cat((spikes[:,None],state.history[:,:-1]),1)
            state.history.copy_(torch.where(v[:,None],history,state.history))
            state.events.add_(spikes);state.ticks.add_(active.to(self.dtype));byte_events.add_(spikes)
            if record:traces.append(spikes.clone())
        return {'features':self.features(state),'events':byte_events,'spikes':traces}

    def features(self,state):
        return self.cfg.readout_gain*(state.rate-self.cfg.target_rate)

    def logits(self,features,level):
        part=features[:,level*self.cfg.N:(level+1)*self.cfg.N]
        return torch.einsum('bn,hvn->bhv',part,self.E)+self.E_bias

    @torch.no_grad()
    def feedback(self,features,target,valid):
        """Negative local CE derivative with respect to filtered spike rates.

        This exact readout derivative constructs desired postsynaptic events;
        it is NOT a derivative of recurrent spike histories with respect to W.
        """
        c=self.cfg
        weights=valid.to(self.dtype)/valid.sum(-1,keepdim=True).clamp_min(1)
        m=torch.zeros_like(features)
        onehot=F.one_hot(target,256).to(self.dtype)
        for l in range(c.L):
            error=(onehot-self.logits(features,l).softmax(-1))*weights[:,:,None]/c.L
            m[:,l*c.N:(l+1)*c.N]=c.readout_gain*torch.einsum('bhv,hvn->bn',error,self.E)
        return m

    @torch.no_grad()
    def teaching_signal(self,features,target,valid):
        c=self.cfg;active=valid.any(-1);n=active.sum().clamp_min(1)
        weights=valid.to(self.dtype)/valid.sum(-1,keepdim=True).clamp_min(1)
        modulator=torch.zeros_like(features);dE=torch.zeros_like(self.E);db=torch.zeros_like(self.E_bias)
        loss=0.
        for l in range(c.L):
            sl=slice(l*c.N,(l+1)*c.N)
            lp=self.logits(features,l).log_softmax(-1)
            error=(F.one_hot(target,256).to(self.dtype)-lp.exp())*weights[:,:,None]/c.L
            modulator[:,sl]=c.readout_gain*torch.einsum('bhv,hvn->bn',error,self.E)
            dE+=torch.einsum('bhv,bn->hvn',error,features[:,sl])/n
            db+=error.sum(0)/n
            loss+=float((-(lp.gather(-1,target[:,:,None]).squeeze(-1)*weights).sum(-1))[active].mean())/c.L
        return modulator,dE,db,loss

    @torch.no_grad()
    def input_pullback(self,signal):
        e=self.E[0] if self.cfg.tie_input else self.E_in
        raw=e.norm(dim=-1,keepdim=True)
        norm=raw.clamp_min(1e-8);unit=e/norm
        g=signal.T
        correction=torch.where(raw>=1e-8,unit*(g*unit).sum(-1,keepdim=True),torch.zeros_like(g))
        return self.cfg.input_gain*(g-correction)/norm

    @torch.no_grad()
    def train_batch(self,batch):
        c=self.cfg;b=batch.to(self.dev);state=self.init_state(len(b.x),learning=True)
        W=self.S+self.A;table=self.input_weights();end=doc_end(b)
        accum={n:torch.zeros_like(p) for n,p in self.params.items()}
        head_signal=torch.zeros_like(self.E[0]);input_signal=torch.zeros_like(self.E[0])
        ticks=0;losses=[];activity=torch.zeros(c.D,device=self.dev,dtype=self.dtype);activity_ticks=0
        for t in range(b.T-1):
            active=b.active[:,t]
            if not bool(active.any()):continue
            y,v=targets(b.x,t,c.horizons,b.P,end);v&=active[:,None]
            teacher=(lambda features:self.feedback(features,y,v)) if t>=b.P-1 and bool(v.any()) and state.teacher is not None else None
            out=self.tick(state,b.x[:,t],active,weights=W,input_weights=table,teacher_signal=teacher)
            activity+=out['events'].sum(0);activity_ticks+=int(active.sum())*c.hops
            if t<b.P-1:continue
            if not bool(v.any()):continue
            m,dE,db,loss=self.teaching_signal(out['features'],y,v)
            accum['E']+=dE;accum['E_bias']+=db
            head_signal+=dE[0]
            supervised=v.any(-1)
            if state.stdp is not None:
                if c.learning=='resume':
                    ones=torch.ones_like(m)
                    direction=state.stdp.modulated(ones,supervised)/c.teacher_gain
                    if c.rule=='no_eligibility':direction.zero_()
                else:direction=state.stdp.modulated(m,supervised)
                if c.rule=='reverse':direction=-direction
                accum['S']+=project(direction,1,self.mask)
                accum['A']+=project(direction,-1,self.mask)
            if state.input_stdp is not None:
                if c.learning=='resume':
                    ones=torch.ones_like(m[:,:c.N])
                    inp=state.input_stdp.modulated(ones,supervised)[0]/c.teacher_gain
                else:inp=state.input_stdp.modulated(m[:,:c.N],supervised)[0]
                if c.rule=='reverse':inp=-inp
                if c.rule=='no_eligibility':inp.zero_()
                inp=self.input_pullback(inp)
                if c.tie_input:
                    accum['E'][0]+=c.input_credit*inp
                    input_signal+=c.input_credit*inp
                else:accum['E_in']+=inp
            ticks+=1;losses.append(loss);self.seen_targets+=int(v.sum())
        if not ticks:return {'no_update':True}
        stats={'loss_nats':sum(losses)/ticks,'spike_rate_by_level':(activity/max(activity_ticks,1)).view(c.L,c.N).mean(-1).tolist(),
               'silent_fraction':float((state.events.sum(0)==0).to(self.dtype).mean()),
               'eligibility_norm':float(state.stdp.eligibility.norm()) if state.stdp else 0.,
               'signal_by_level':{},'weight_change_by_level':{}}
        stats['dictionary_signals']={'readout_norm':float(head_signal.norm()/ticks),
            'tied_input_norm':float(input_signal.norm()/ticks),
            'input_to_readout_ratio':float(input_signal.norm()/head_signal.norm().clamp_min(1e-12)),
            'cosine':float(F.cosine_similarity(head_signal.flatten(),input_signal.flatten(),dim=0))}
        before={n:getattr(self,n).clone() for n in ('S','A')}
        for n in ('S','A'):
            stats['signal_by_level'][n]=[float(accum[n][:,l*c.N:(l+1)*c.N,l*c.N:(l+1)*c.N].norm()/ticks) for l in range(c.L)]
        if self.optimizer:
            owned={id(p) for group in self.optimizer.param_groups for p in group['params']}
            for n,p in self.params.items():p.grad=-accum[n]/ticks if id(p) in owned else None
            self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
        self.S.copy_(project(self.S,1,self.mask));self.A.copy_(project(self.A,-1,self.mask))
        if c.homeostasis:self.theta.add_(c.homeostasis*(activity/max(activity_ticks,1)-c.target_rate)).clamp_(.2,3.)
        for n in ('S','A'):
            stats['weight_change_by_level'][n]=[float((getattr(self,n)-before[n])[:,l*c.N:(l+1)*c.N,l*c.N:(l+1)*c.N].norm()) for l in range(c.L)]
        for n in self.params:
            if not bool(torch.isfinite(getattr(self,n)).all()):raise FloatingPointError(n)
        self.updates+=1
        return stats

    @torch.no_grad()
    def checkpoint(self):
        return {'version':1,'dynamics':'event-stdp-rrem-20260923','config':asdict(self.cfg),
                'parameters':{n:getattr(self,n).clone() for n in (*self.params,'theta','tonic','leak','initial_u')},
                'optimizer':copy.deepcopy(self.optimizer.state_dict()) if self.optimizer else None,
                'updates':self.updates,'seen_targets':self.seen_targets}

    @classmethod
    def from_checkpoint(cls,ck,device=None):
        if ck.get('dynamics')!='event-stdp-rrem-20260923' or ck.get('version')!=1:raise ValueError('incompatible spiking checkpoint')
        cfg=dict(ck['config']);cfg['stdp']=STDPConfig(**cfg['stdp'])
        if device is not None:cfg['device']=device
        m=cls(SpikeConfig(**cfg))
        with torch.no_grad():
            for n,p in ck['parameters'].items():getattr(m,n).copy_(p.to(m.dev))
        if m.optimizer:m.optimizer.load_state_dict(ck['optimizer'])
        m.updates=ck['updates'];m.seen_targets=ck['seen_targets']
        return m


@torch.no_grad()
def evaluate_spiking(m,batches,*,per_document=False,reset_history=False):
    c=m.cfg
    totals=torch.zeros(c.L,c.horizons,device=m.dev,dtype=torch.float64);counts=torch.zeros(c.horizons,device=m.dev,dtype=torch.float64)
    documents=[];activity=torch.zeros(c.D,device=m.dev,dtype=m.dtype);nticks=0
    W=m.S+m.A;table=m.input_weights()
    for batch in batches:
        b=batch.to(m.dev);state=m.init_state(len(b.x));end=doc_end(b)
        dc=torch.zeros(len(b.x),c.L,c.horizons,device=m.dev,dtype=torch.float64)
        dn=torch.zeros(len(b.x),c.horizons,device=m.dev,dtype=torch.float64)
        for t in range(b.T-1):
            act=b.active[:,t]
            if not bool(act.any()):continue
            if reset_history:state=m.init_state(len(b.x))
            out=m.tick(state,b.x[:,t],act,weights=W,input_weights=table)
            activity+=out['events'].sum(0);nticks+=int(act.sum())*c.hops
            if t<b.P-1:continue
            y,v=targets(b.x,t,c.horizons,b.P,end);v&=act[:,None];dn+=v
            for l in range(c.L):
                lp=m.logits(out['features'],l).log_softmax(-1)
                dc[:,l]+=(-lp.gather(-1,y[:,:,None]).squeeze(-1)*v).double()
        totals+=dc.sum(0);counts+=dn.sum(0)
        if per_document:
            documents.extend({'id':int(i),'nll_sum':v,'counts':n} for i,v,n in zip(b.doc_ids,dc.cpu().tolist(),dn.cpu().tolist()))
    bpb=totals/counts.clamp_min(1)/math.log(2)
    return {'bpb':bpb.tolist(),'bpb_h1':float(bpb[-1,0]),'bpb_mean_all_h':float(bpb[-1].mean()),
            'counts':counts.tolist(),'spike_rate_by_level':(activity/max(nticks,1)).view(c.L,c.N).mean(-1).tolist(),
            **({'documents':documents} if per_document else {})}
