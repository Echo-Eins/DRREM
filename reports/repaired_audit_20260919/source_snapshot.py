"""RREM: experimental finite-horizon, bidirectional byte machine.

Revision of the user's 2026-09-19 rrem.py, NOT a claim of language-model success.
No equilibrium constraint, spectral cap, fixed-point solver, BPTT, or autograd
is used by training. Every level predicts all eight offsets. All adjacent and
within-level directed connections remain available.

Learning contract
-----------------
The main rule differentiates local predictive losses through the emitting neuron
and its *passive membrane integration*, holding communicated messages, traces,
and adaptation histories fixed. This is a precisely defined local surrogate,
NOT the exact gradient through the full recurrent computation. Its presynaptic
eligibility is forward-only; it does not solve arbitrary long-range credit.
FF contributes separate positive and negative local derivatives. S/A updates
are matrix projections, not a theorem that time parity equals semantic function.

Changes: evaluation is read-only; true untied-input option; separate content and
excitability; non-exhaustive divisive budget; hop-independent oscillator period;
correct phase derivative; all-hop/all-horizon local supervision; gate derivative
includes the actual synapse; corrected temperature and tied-input derivatives;
signal-proportional optimizer (optional norm-restored NS direction); complete
batch-boundary checkpoints; explicit data/test separation.

Standalone import requires only torch. CLI uses the original drrem data loader
lazily, or explicit JSONL files containing real OpenOrca rows. No dataset is
manufactured or downloaded by this module.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass
class Cfg:
    N: int = 256
    L: int = 2
    H_pred: int = 8
    hops: int = 8
    trace_taus: tuple[float, ...] = (2., 8., 32., 128.)
    delay_lags: tuple[int, ...] = (1, 2, 3, 4)
    neuron_mode: str = 'separate'  # separate or legacy: original forward dynamics as an ablation
    alpha: float = .5
    theta0: float = 0.
    beta_a: float = .5
    tau_a: float = 8.             # observed-byte ticks
    tau_ref: float = 1.5          # internal hops, unlike the original
    beta_ref: float = 1.
    homeo_rate: float = .001
    target_act: float = .15
    content_T: float = 1.
    route_T: float = 1.
    route_budget: float = 1.
    use_phase: bool = True
    phase_depth: float = .5       # psi in [1-depth, 1], never reverses sign
    phase_period: float = 8.      # independent of inference budget
    gamma_A: float = .25   # при γ=1 выравнивание правила с градиентом падает 0,994 → 0,865 — худший измеренный режим
    g_S: float = .4
    g_A: float = .4
    gate_init: float = 1.
    symmetric_gate: bool = True  # preserves S*g symmetric and A*g skew
    g_r: float = 1.
    tau_r: float = 1.
    tie_input: bool = True
    in_gain: float = 8.
    # These ARE optimizer hyperparameters; removing their names does not remove them.
    zeta: float = .02
    head_scale: float = 1.
    # Саморегулирующийся темп ПО БЛОКАМ, онлайн, без отложенных данных: для каждого блока (своя
    # читалка уровня, строки S/A/вентилей этого уровня) по двум последним шагам оценивается обратная
    # кривизна Барзилая–Борвейна  η = ⟨Δp, Δg⟩ / ‖Δg‖².  Это ровно тот шаг, который подходит блоку
    # сейчас; нужны только его собственные величины. При ⟨Δp, Δg⟩ ≤ 0 (вдоль шага рельеф не выпуклый)
    # темп блока убавляется. Так у каждого уровня свой темп, и общий ζ перестаёт обслуживать две
    # разные задачи — выпуклую (читалка) и невыпуклую (рекуррентность).
    # Куда цепляется словарь байт. 'last' — схема постановщика: ОДИН словарь на 256 байт, висит
    # только на последнем слое; нижние слои учатся FF и STDP, никакого чтения у них нет и ошибка
    # предсказания в них напрямую не течёт. 'all' — читалка на каждом уровне (было раньше).
    readout_levels: str = 'last'
    # ВНИМАНИЕ: в прогонах 250 батчей эта самонастройка ухудшила результат (4,903 против 4,630 на
    # том же шаге): оценка want смешивает норму направления с нормой сигнала и срезает темп читалки
    # вчетверо. Идея верна, реализация нет — по умолчанию выключено, пока не будет выведена честно.
    # README §8–9: R_h = Δlog p − λ₀ − λ_s·N_событий − λ_e·N_рёбер модулирует обновление. В присланной
    # ветке R считался и выбрасывался: изменение λ₀ с 0 на 10 давало 0,0 разницы по всем семи
    # градиентам, то есть цена маршрута и вентили не обучались вовсе.
    # README §11: g_ij(t) = g(z_t, m_t) — вентиль зависит от СОСТОЯНИЯ, а не статическая матрица.
    # Факторизованная форма g_ij = g_i·g_j (она же §14: message = g(r_i,r_j)·c_i) даёт настоящее
    # произведение «сигнал × функция состояния», то есть конъюнкцию, но стоит O(D), а не O(D²) памяти.
    # Считается по состоянию ПРЕДЫДУЩЕГО хопа, поэтому входит в правило записанной константой —
    # тем же статусом, что уже имеют сообщения, и контракт локальной производной не меняется.
    # Обусловленность. Двойник старой машины на истинном градиенте дал 2,666 с Adam и только 3,535
    # на чистом SGD — то есть 0,87 бита принесла именно по-координатная нормировка. Прежний довод
    # «Adam на локальном сигнале проваливается» относился к СМЕЩЁННОМУ сигналу (cos 0,88, согласие
    # знаков 0,76): он раздувал шум в мелких координатах. Наш сигнал на такте точен (cos 1,0000),
    # и этот довод к нему не применяется — проверяется отдельной веткой.
    # Задача слоя: L_ℓ = CE(горизонты MTP) + λ_E·E_ℓ + λ_C·цена маршрута.
    #   CE говорит, КАКИЕ состояния правильные; энергия делает их устойчивыми; цена — дешёвыми.
    #   Энергия по README §18 живёт только в симметричной части: E = ΣΦ(u) − ½uᵀSu − uᵀI,
    #   значит −∂E/∂S = u uᵀ. Одна она вырождена (любое состояние станет аттрактором), поэтому
    #   идёт строго в паре с CE. Антисимметричная A в энергию не входит и учится CE и таймингом.
    lam_energy: float = 0.
    optimizer: str = 'muon'     # 'muon' — ортогонализация направления (приоритет); 'adam'; 'signal'
    adam_betas: tuple = (.9,.999)
    adam_eps: float = 1e-8
    gate_state: float = 0.
    # README §5: «активируйся, если A было недавно, B чуть раньше» — момент срабатывания зависит ОТ ВХОДА.
    # Сдвиг фазы пропорционален собственной раскачке нейрона: сильнее ведут — раньше говорит.
    phase_input: float = 0.
    reward_weight: float = .5
    carry_elig: bool = False   # продолжать след годности через такты (e-prop) вместо сброса на байте
    bb: bool = False
    bb_ema: float = .5
    bb_lo: float = .05
    bb_hi: float = 20.
    gate_scale: float = 1.
    phase_scale: float = 1.
    elig_decay: float = .95       # optimizer momentum, NOT neural eligibility
    muon: bool = False           # optional NS direction with restored signal norm
    muon_iters: int = 5
    oja: float = 0.              # optional activity damping; NOT part of CE gradient
    freeze: tuple[str, ...] = () # W, gate, E, E_in, phi, theta
    hop_loss: str = 'all'       # all or last: auxiliary supervision is an ablation, not a requirement
    ff_weight: float = .03
    ff_every: int = 4
    ff_theta: float = .1         # below max mean(tanh(u)^2)=1
    lam_hop: float = 0.
    lam_spike: float = 0.        # differentiable squared-message cost, not event count
    lam_edge: float = 0.         # effective squared synaptic-current proxy, NOT hardware edge count
    pace: bool = False           # optional development-set hyperparameter adaptation
    pace_strength: float = .05
    seed: int = 20260919
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: str = 'float32'

    @property
    def D(self) -> int: return self.N * self.L
    @property
    def M(self) -> int: return 1 + len(self.trace_taus) + len(self.delay_lags)

    def validate(self) -> None:
        if min(self.N,self.L,self.H_pred,self.hops) < 1:
            raise ValueError('N, L, H_pred, hops must be positive')
        if not 0 < self.alpha <= 1: raise ValueError('alpha must lie in (0, 1]')
        if not 0 <= self.phase_depth <= 1: raise ValueError('phase_depth must lie in [0,1]')
        if not 0 <= self.gate_init <= 1: raise ValueError('gate_init must lie in [0,1]')
        if not 0 <= self.elig_decay < 1: raise ValueError('elig_decay must lie in [0,1)')
        if min(self.content_T,self.route_T,self.route_budget,self.tau_r,self.tau_a,self.tau_ref,self.phase_period) <= 0:
            raise ValueError('temperatures, time constants, budget and period must be positive')
        if any(t <= 0 for t in self.trace_taus): raise ValueError('trace taus must be positive')
        if any(not isinstance(t,int) or t < 1 for t in self.delay_lags):
            raise ValueError('delay_lags must contain positive integers')
        if self.dtype not in ('float32','float64'): raise ValueError('use float32 or float64')
        if self.tie_input and 'E_in' in self.freeze:
            raise ValueError('For frozen-input readout probing set tie_input=False first')
        if not 0 <= self.ff_theta < 1: raise ValueError('FF threshold must lie in [0,1)')
        if min(self.zeta,self.ff_weight,self.oja,self.lam_hop,self.lam_spike,self.lam_edge) < 0:
            raise ValueError('learning scales and costs must be nonnegative')
        if self.ff_every < 1: raise ValueError('ff_every must be positive')
        if self.hop_loss not in ('all','last'):raise ValueError('hop_loss must be all or last')
        if self.neuron_mode not in ('separate','legacy'):raise ValueError('neuron_mode must be separate or legacy')
        allowed={'W','gate','E','E_in','phi','theta'}
        if set(self.freeze)-allowed:raise ValueError(f'unknown frozen groups: {set(self.freeze)-allowed}')
        if min(self.head_scale,self.gate_scale,self.phase_scale,self.homeo_rate) < 0:raise ValueError('scales must be nonnegative')


@dataclass
class State:
    u: torch.Tensor
    a: torch.Tensor
    ref: torch.Tensor
    traces: torch.Tensor
    delays: torch.Tensor
    msg: torch.Tensor
    elig: torch.Tensor | None = None   # след годности, продолжающийся ЧЕРЕЗ такты (e-prop)
    p_prev: torch.Tensor | None = None

    def detach_clone(self) -> 'State':
        def c(t): return None if t is None else t.detach().clone()
        return State(*(c(getattr(self,f.name)) for f in fields(self)))


def project(x: torch.Tensor, sym: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    if sym: x = .5 * (x + sym*x.transpose(-1,-2))
    return x if mask is None else x*mask


@torch.no_grad()
def orthogonalize(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Approximate NS polar direction, not an exact orthogonal matrix."""
    norm=x.norm()
    if float(norm)==0: return torch.zeros_like(x)
    y=x/norm
    for _ in range(iters):
        a=y@y.T
        y=3.4445*y + (-4.7750*a + 2.0315*(a@a))@y
    return y


class SignalRate:
    """Batch-mean momentum and row coherence; neural eligibility lives in tick().

    NS, when selected, changes direction ONLY: its norm is restored to that of
    the coherent mean signal. Small coherent signals therefore make small steps.
    Coherence is a descriptive statistic, not a guarantee of correct gradients.
    """
    def __init__(self,p: torch.Tensor,rho: float):
        self.rho=rho; self.mean=torch.zeros_like(p)
        self.power=torch.zeros_like(p[...,0]) if p.ndim>1 else torch.zeros_like(p)
        self.steps=0

    @torch.no_grad()
    def direction(self,g: torch.Tensor,*,sym: int=0,mask=None,muon=False,iters=5):
        self.steps+=1
        self.mean.mul_(self.rho).add_(g,alpha=1-self.rho)
        power=g.square().sum(-1) if g.ndim>1 else g.square()
        self.power.mul_(self.rho).add_(power,alpha=1-self.rho)
        corr=1-self.rho**self.steps
        m=self.mean/corr; v=self.power/corr
        n=m.square().sum(-1) if g.ndim>1 else m.square()
        conf=(n/v.clamp_min(torch.finfo(g.dtype).tiny)).clamp(0,1).sqrt()
        # Шумовой пол: у ρ-затухающего накопителя чистый шум даёт conf = √((1−ρ)/(1+ρ)), а не ноль.
        # Без вычитания пола строка со случайным сигналом идёт полным ходом (измерено: 0,159 против
        # 0,006 на 400 шагах чистого шума). Вычитаем и перенормируем — тогда шум действительно стоит.
        floor=math.sqrt((1-self.rho)/(1+self.rho))
        conf=((conf-floor)/(1-floor)).clamp_min(0)
        if g.ndim==1: d=m*conf
        elif sym:
            d=m*conf.sqrt().unsqueeze(-1)*conf.sqrt().unsqueeze(-2)
        else: d=m*conf.unsqueeze(-1)
        d=project(d,sym,mask)
        if muon and d.ndim>=2:
            flat=d.reshape(-1,d.shape[-2],d.shape[-1]); result=[]
            for a in flat:
                q=orthogonalize(a,iters)
                q=project(q,sym,mask)
                q=q*(a.norm()/q.norm().clamp_min(torch.finfo(q.dtype).tiny))
                result.append(q)
            d=torch.stack(result).reshape_as(d)
        return d, float(conf.mean())

    def state_dict(self):
        return {'mean':self.mean,'power':self.power,'steps':self.steps,'rho':self.rho}

    def load_state_dict(self,d):
        self.mean.copy_(d['mean']);self.power.copy_(d['power']);self.steps=int(d['steps'])
        self.rho=float(d['rho'])


class RREM:
    param_names=('S','A','gate','E','E_bias','E_in','phi')

    def __init__(self,cfg: Cfg):
        cfg.validate();self.cfg=cfg;self.dev=torch.device(cfg.device)
        self.dtype=getattr(torch,cfg.dtype);N,L,D,M=cfg.N,cfg.L,cfg.D,cfg.M
        gen=torch.Generator().manual_seed(cfg.seed)
        rnd=lambda *shape:torch.randn(*shape,generator=gen,dtype=self.dtype).to(self.dev)
        lvl=torch.arange(D,device=self.dev)//N
        self.mask=((lvl[:,None]-lvl[None,:]).abs()<=1).to(self.dtype)
        sc=1/math.sqrt(N*M)
        self.S=project(rnd(M,D,D)*cfg.g_S*sc,1,self.mask)
        self.A=project(rnd(M,D,D)*cfg.g_A*sc,-1,self.mask)
        self.gate=torch.full((D,D),cfg.gate_init,device=self.dev,dtype=self.dtype)*self.mask
        # СВОЯ читалка на каждый уровень. Общая E принуждает все уровни быть декодируемыми одной
        # линейной картой, то есть прямо тянет их к одному представлению и обнуляет смысл иерархии.
        # Измерено на замороженной машине: своя читалка на уровень 3,701 против 3,829 у общей.
        self.E=rnd(cfg.L,cfg.H_pred,256,N)*cfg.g_r/math.sqrt(N)
        self.E_bias=torch.zeros(cfg.L,cfg.H_pred,256,device=self.dev,dtype=self.dtype)
        self.E_in=self.E[0,0].clone()  # active only if tie_input=False
        self.phi=(torch.rand(D,generator=gen,dtype=self.dtype)*2*math.pi).to(self.dev)
        self.theta=torch.full((D,),cfg.theta0,device=self.dev,dtype=self.dtype)
        self.decay=torch.tensor([math.exp(-1/t) for t in cfg.trace_taus],device=self.dev,dtype=self.dtype)
        self.max_lag=max(cfg.delay_lags,default=0)
        self.read_levels=tuple(range(cfg.L)) if cfg.readout_levels=='all' else (cfg.L-1,)
        self.rates={name:SignalRate(getattr(self,name),cfg.elig_decay) for name in self.param_names}
        self.zeta=cfg.zeta;self.updates=0;self.seen_targets=0
        self.bb={};self.adam={}
        self.grad={name:torch.zeros_like(getattr(self,name)) for name in self.param_names}
        self.grad_ticks=0
        self.act_sum=torch.zeros(D,device=self.dev,dtype=self.dtype);self.act_count=0

    def init_state(self,B: int):
        z=lambda *shape:torch.zeros(*shape,device=self.dev,dtype=self.dtype)
        D=self.cfg.D
        return State(z(B,D),z(B,D),z(B,D),z(B,len(self.cfg.trace_taus),D),
                     z(B,self.max_lag,D),z(B,D),z(B,self.cfg.M,D))

    def channels(self,st: State):
        if not self.cfg.delay_lags:return st.traces
        d=torch.stack([st.delays[:,l-1] for l in self.cfg.delay_lags],1)
        return torch.cat((st.traces,d),1)

    def W(self): return (self.S+self.cfg.gamma_A*self.A)*self.gate[None]

    def input_drive(self,byte: torch.Tensor):
        if byte.dtype!=torch.long or bool(((byte<0)|(byte>255)).any()):
            raise ValueError('byte IDs must be torch.long in [0,255]')
        e=self.E[0,0] if self.cfg.tie_input else self.E_in
        row=e[byte];norm=row.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        I=torch.zeros(byte.shape[0],self.cfg.D,device=self.dev,dtype=self.dtype)
        I[:,:self.cfg.N]=self.cfg.in_gain*row/norm
        return I

    def emitter(self,u: torch.Tensor,theta: torch.Tensor,hop: int,phase_shift=None):
        """Signed content is separate from nonnegative excitability.

        Inhibition reduces absolute communication rather than pushing content
        toward -1. Divisive normalization bounds maximum routing without forcing
        the entire budget to be spent when every neuron should be silent.
        """
        cfg=self.cfg;B=u.shape[0];shape=(B,cfg.L,cfg.N)
        if cfg.neuron_mode=='legacy':
            z=(u-theta)/cfg.route_T;content=torch.tanh(z);q=F.softplus(z).view(shape)
            den=q.norm(dim=-1,keepdim=True).clamp_min(1e-9)
            route=(cfg.route_budget*math.sqrt(cfg.N)*q/den).reshape_as(u)
            if cfg.use_phase:
                angle=2*math.pi*hop/cfg.hops-self.phi
                psi=1-cfg.phase_depth*(1-torch.cos(angle))
            else:psi=torch.ones_like(self.phi)
            return content*route*psi,{'u':u,'theta':theta,'content':content,'z':z,'q':q,'den':den,
                                     'route':route,'psi':psi,'hop':hop}
        content=torch.tanh(u/cfg.content_T)
        # Smooth absolute value, preserving a zero reference at u=0.
        eps=1e-4
        mag=(u.square()+eps**2).sqrt()-eps
        z=(mag-theta)/cfg.route_T
        q=F.softplus(z).view(shape)
        den=(1+q.square().mean(-1,keepdim=True)).sqrt()
        route=(cfg.route_budget*q/den).reshape_as(u)
        if cfg.use_phase:
            ang=2*math.pi*hop/cfg.phase_period-self.phi
            if phase_shift is not None: ang=ang-phase_shift
            psi=1-.5*cfg.phase_depth+.5*cfg.phase_depth*torch.cos(ang)
        else: psi=torch.ones_like(self.phi)
        msg=content*route*psi
        return msg,{'u':u,'theta':theta,'content':content,'z':z,'q':q,'den':den,'route':route,'psi':psi,
                    'hop':hop,'ang':ang if cfg.use_phase else None}

    def emit_vjp(self,cache: dict,signal: torch.Tensor):
        """Analytic local negative-loss signal to u and phase; no graph traversal."""
        cfg=self.cfg;u=cache['u'];c=cache['content'];r=cache['route'];psi=cache['psi']
        B=u.shape[0];q=cache['q'];den=cache['den']
        v=(signal*c*psi).view(B,cfg.L,cfg.N)
        if cfg.neuron_mode=='legacy':
            unclamped=q.norm(dim=-1,keepdim=True)>1e-9
            dq=cfg.route_budget*math.sqrt(cfg.N)*(v/den-q*(v*q).sum(-1,keepdim=True)/den.pow(3)*unclamped)
            du=(signal*r*psi*(1-c.square())+dq.reshape_as(u)*torch.sigmoid(cache['z']))/cfg.route_T
            if cfg.use_phase:
                angle=2*math.pi*cache['hop']/cfg.hops-self.phi
                dp=signal*c*r*cfg.phase_depth*torch.sin(angle)
            else:dp=torch.zeros_like(u)
            return du,dp
        dq=cfg.route_budget*(v/den-q*(v*q).mean(-1,keepdim=True)/den.pow(3))
        mag_grad=u/(u.square()+1e-8).sqrt()
        du=signal*r*psi*(1-c.square())/cfg.content_T
        du=du+dq.reshape_as(u)*torch.sigmoid(cache['z'])*mag_grad/cfg.route_T
        if cfg.use_phase:
            ang=cache['ang'] if cache.get('ang') is not None else 2*math.pi*cache['hop']/cfg.phase_period-self.phi
            dphi=(signal*c*r)*(.5*cfg.phase_depth*torch.sin(ang))
        else:dphi=torch.zeros_like(u)
        return du,dphi

    def tick(self,st: State,I: torch.Tensor,learn: bool=False,*,hops: int|None=None):
        """No target is accepted here. States therefore cannot leak future labels."""
        H=self.cfg.hops if hops is None else hops
        if H<1:raise ValueError('hops must be positive')
        cfg=self.cfg;W=self.W();ch=self.channels(st)
        slow=torch.einsum('bmi,mji->bj',ch,W[1:]) if cfg.M>1 else torch.zeros_like(st.u)
        pre,u,ref=st.msg,st.u,st.ref
        # Ключевая развилка присвоения заслуг. Обнуляя след на границе байта, правило объявляет всё
        # переносимое (мембрану, сообщение, следы, задержки) константой — и тогда оно точно лишь для
        # ОДНОГО такта: измерено cos 0,996 при одном такте, 0,425 при восьми, 0,342 при шестнадцати.
        # Продолжая след с собственной утечкой нейрона, получаем e-prop: та же трёхфакторная форма,
        # тот же локальный сигнал, но косвенный путь через историю больше не теряется. У этой машины
        # всё переносимое — по-нейронное, поэтому отбрасывается только то, что ушло к соседям и вернулось.
        elig=(st.elig if (cfg.carry_elig and st.elig is not None)
              else torch.zeros(st.u.shape[0],cfg.M,cfg.D,device=self.dev,dtype=self.dtype))
        inp_coeff=0.;msgs=[];routes=[];records=[]
        ref_decay=math.exp(-1/cfg.tau_ref)
        for k in range(H):
            # All messages in this recorded path are constants to the LOCAL rule.
            theta=self.theta[None]+cfg.beta_a*st.a+cfg.beta_ref*ref
            # Раскачка по состоянию ПРЕДЫДУЩЕГО хопа: и вентиль, и сдвиг фазы берутся отсюда,
            # поэтому для локального правила это записанные константы, как и сообщения.
            zprev=(u-theta)/cfg.route_T
            gd=torch.sigmoid(cfg.gate_state*zprev) if cfg.gate_state>0 else None
            pre_eff=pre*gd if gd is not None else pre          # исходящий множитель g_j
            channels=torch.cat((pre_eff[:,None],ch),1)
            rec=pre_eff@W[0].T+slow
            if gd is not None: rec=rec*gd                      # входящий множитель g_i ⇒ g_ij = g_i·g_j
            field=rec+I-st.a   # README §5: u ← αu + Σ w x − a; вентиль не трогает внешний вход и адаптацию
            u=(1-cfg.alpha)*u+cfg.alpha*field
            msg,cache=self.emitter(u,theta,k,
                                   phase_shift=cfg.phase_input*zprev if cfg.phase_input>0 else None)
            if gd is not None: cache['gdyn']=gd
            elig=(1-cfg.alpha)*elig+cfg.alpha*channels
            inp_coeff=(1-cfg.alpha)*inp_coeff+cfg.alpha
            if learn:
                cache['elig']=elig;cache['input_coeff']=inp_coeff;cache['pre']=pre
                records.append(cache)
            msgs.append(msg);routes.append(cache['route'])
            if cfg.neuron_mode!='legacy':ref=ref_decay*ref+(1-ref_decay)*msg.abs()
            pre=msg
        if cfg.neuron_mode=='legacy':ref=ref_decay*ref+(1-ref_decay)*msg.abs()
        if self.cfg.carry_elig:st.elig=elig.detach()
        return {'u':u,'ref':ref,'msgs':msgs,'routes':routes,'ch':ch,'records':records}

    def logits(self,msg: torch.Tensor,level: int):
        z=msg.view(-1,self.cfg.L,self.cfg.N)[:,level]
        return torch.einsum('bn,hvn->bhv',z,self.E[level])/self.cfg.tau_r+self.E_bias[level]

    @torch.no_grad()
    def advance(self,st: State,out: dict,valid: torch.Tensor,*,plastic: bool=False):
        """Causal state changes always; shared homeostasis only collected on train answers."""
        v=valid[:,None];msg=out['msgs'][-1]
        st.u=torch.where(v,out['u'],st.u);st.msg=torch.where(v,msg,st.msg)
        st.ref=torch.where(v,out['ref'],st.ref)
        decay=math.exp(-1/self.cfg.tau_a)
        st.a=torch.where(v,decay*st.a+(1-decay)*msg.abs(),st.a)
        st.traces=torch.where(v[:,None],self.decay[None,:,None]*st.traces+(1-self.decay[None,:,None])*msg[:,None],st.traces)
        if self.max_lag:
            nxt=torch.cat((msg[:,None],st.delays[:,:-1]),1)
            st.delays=torch.where(v[:,None],nxt,st.delays)
        if plastic and bool(valid.any()):
            self.act_sum+=msg[valid].abs().sum(0);self.act_count+=int(valid.sum())

    def _input_update(self,dI: torch.Tensor,byte: torch.Tensor):
        name='E' if self.cfg.tie_input else 'E_in'
        e=self.E[0,0] if self.cfg.tie_input else self.E_in
        row=e[byte];norm=row.norm(dim=-1,keepdim=True)
        norm=norm.clamp_min(1e-8);unit=row/norm
        signal=dI[:,:self.cfg.N]
        dr=self.cfg.in_gain*(signal-unit*(signal*unit).sum(-1,keepdim=True))/norm
        if self.cfg.tie_input:self.grad['E'][0,0].index_add_(0,byte,dr)
        else:self.grad[name].index_add_(0,byte,dr)

    def _field_update(self,du: torch.Tensor,cache: dict):
        # This is the forward eligibility × postsynaptic learning signal rule.
        # При состояние-зависимом вентиле эффективная связь есть g_i·gate⁰_ij·g_j·W_ij, поэтому
        # множитель g_i входит в сигнал поста, а g_j уже сидит в следе (он копит pre_eff).
        if 'gdyn' in cache: du=du*cache['gdyn']
        K=torch.einsum('bi,bmj->mij',du,cache['elig'])
        gated=K*self.gate[None]
        self.grad['S']+=project(gated,1,self.mask)
        self.grad['A']+=self.cfg.gamma_A*project(gated,-1,self.mask)
        self.grad['gate']+=((K*(self.S+self.cfg.gamma_A*self.A)).sum(0))*self.mask

    def level_energy(self,msg: torch.Tensor,I: torch.Tensor):
        """Энергия уровня (README §18): E_ℓ = ΣΦ(u) − ½ uᵀ S u − uᵀ I, где внутри уровня берётся его
        собственный блок S, а вклад соседних уровней и внешнего входа учитывается как поле. (B, L)"""
        cfg=self.cfg;B=msg.shape[0];u=msg.view(B,cfg.L,cfg.N)
        S0=(self.S*self.gate[None]).sum(0)   # эффективная симметричная связь по всем каналам
        quad=torch.stack([(u[:,l]*(u[:,l]@S0[l*cfg.N:(l+1)*cfg.N,l*cfg.N:(l+1)*cfg.N].T)).sum(-1)
                          for l in range(cfg.L)],1)
        field=(msg*I).view(B,cfg.L,cfg.N).sum(-1)
        return .5*u.square().sum(-1)-.5*quad-field

    def energy_grad_S(self,msg: torch.Tensor):
        """−∂E/∂S = u uᵀ внутри каждого уровня: хеббовский член, делающий текущее состояние устойчивым.
        Между уровнями не пишем — это поле соседа, а не собственная энергия уровня."""
        cfg=self.cfg;B=msg.shape[0];out=torch.zeros_like(self.S[0])
        for l in range(cfg.L):
            sl=slice(l*cfg.N,(l+1)*cfg.N);z=msg[:,sl]
            out[sl,sl]=.5*(z.T@z)/B   # −∂(−½uᵀSu)/∂S = ½ u uᵀ (проверено против автограда, cos 1,0)
        return out

    def goodness(self,cache: dict):
        return cache['content'].view(-1,self.cfg.L,self.cfg.N).square().mean(-1)

    @torch.no_grad()
    def learn_tick(self,st: State,out: dict,byte: torch.Tensor,Y: torch.Tensor,V: torch.Tensor,
                   *,negative: tuple[dict,torch.Tensor]|None=None) -> dict:
        """All-hop, all-level, all-horizon local losses. No target modifies State.

        Each tick is normalized by its number of valid (sample,horizon) labels,
        then by levels and hops. A longer horizon is never given an extra smaller
        weight. End-of-document missing labels are excluded, not fabricated.
        """
        cfg=self.cfg;valid=V.to(self.dtype);nv=float(valid.sum())
        if nv==0:return {'valid_targets':0,'R':0.,'local_loss':float('nan')}
        H=len(out['msgs']);den=nv*cfg.L*H;B=Y.shape[0]
        active=V.any(-1);dI=torch.zeros_like(st.u);costs=[];per_hop=[];loss_sum=0.
        allowed=(cfg.M*self.mask.sum()).clamp_min(1)
        base=self.S+cfg.gamma_A*self.A;effective=base*self.gate[None]
        onehot=F.one_hot(Y,256).to(self.dtype)
        if len(out['records'])!=H:raise ValueError('learn_tick requires tick(..., learn=True)')
        for msg,cache in zip(out['msgs'],out['records']):
            pred_weight=1. if cfg.hop_loss=='all' else float(H if cache['hop']==H-1 else 0.)
            signal=torch.zeros_like(msg);ce_sample=torch.zeros(B,device=self.dev,dtype=self.dtype)
            for l in self.read_levels:
                lg=self.logits(msg,l);lp=lg.log_softmax(-1);p=lp.exp()
                err=(onehot-p)*valid[:,:,None]*(pred_weight/den)
                z=msg.view(B,cfg.L,cfg.N)[:,l]
                self.grad['E'][l]+=torch.einsum('bhv,bn->hvn',err,z)/cfg.tau_r
                self.grad['E_bias'][l]+=err.sum(0)
                signal[:,l*cfg.N:(l+1)*cfg.N]+=torch.einsum('bhv,hvn->bn',err,self.E[l])/cfg.tau_r
                ce=-(lp.gather(-1,Y[:,:,None]).squeeze(-1)*valid)
                loss_sum+=float(ce.sum())*pred_weight/den
                ce_sample+=ce.sum(-1)/valid.sum(-1).clamp_min(1)/len(self.read_levels)
            # Explicit soft communication costs in the local predictive objective.
            w=active.to(self.dtype)/active.sum().clamp_min(1)/H
            # Cost depends on actual g*W transmission, not g alone. Scaling
            # g -> g/c and W -> c*W cannot buy fictitiously cheaper computation.
            channels=torch.cat((cache['pre'][:,None],out['ch']),1)
            edge_power=(channels.square()*effective.square().sum(1)[None]).sum((1,2))/allowed
            cost=cfg.lam_hop+cfg.lam_spike*msg.square().mean(-1)+cfg.lam_edge*edge_power
            signal-=w[:,None]*(2*cfg.lam_spike*msg/cfg.D)
            direct=-2*cfg.lam_edge*effective*(channels.square()*w[:,None,None]).sum(0)[:,None,:]/allowed
            self.grad['S']+=project(direct*self.gate[None],1,self.mask)
            self.grad['A']+=cfg.gamma_A*project(direct*self.gate[None],-1,self.mask)
            self.grad['gate']+=(direct*base).sum(0)*self.mask
            du,dphi=self.emit_vjp(cache,signal)
            if cfg.reward_weight>0 and per_hop:
                # R_h = (польза этого хопа) − (его цена). Связи, участвовавшие в полезной цепочке,
                # получают усиленный кредит, в бесполезной — ослабленный (README §9).
                R=per_hop[-1]-ce_sample-cost
                du=du*(1+cfg.reward_weight*torch.tanh(R))[:,None]
                dphi=dphi*(1+cfg.reward_weight*torch.tanh(R))[:,None]
            self._field_update(du,cache)
            self.grad['phi']+=dphi.sum(0)
            dI+=cache['input_coeff']*du
            costs.append(cost);per_hop.append(ce_sample)
        if cfg.lam_energy>0:
            # понижение энергии уровня: хебб по последнему сообщению такта, через все каналы
            eg=cfg.lam_energy*self.energy_grad_S(out['msgs'][-1])
            self.grad['S']+=project(eg[None].expand_as(self.S)/cfg.M,1,self.mask)
        self._input_update(dI,byte)
        ff_loss=0.
        if negative is not None and cfg.ff_weight>0:
            neg,negative_byte=negative
            if len(neg['records'])!=H:raise ValueError('FF paths must have equal budgets')
            for path,sgn,input_byte in ((out,1.,byte),(neg,-1.,negative_byte)):
                dIp=torch.zeros_like(st.u)
                for cache in path['records']:
                    g=self.goodness(cache)
                    strength=(torch.sigmoid(cfg.ff_theta-g) if sgn>0 else -torch.sigmoid(g-cfg.ff_theta))
                    strength*=active[:,None].to(self.dtype)*cfg.ff_weight/(active.sum().clamp_min(1).to(self.dtype)*cfg.L*H)
                    c=cache['content']
                    ct=cfg.route_T if cfg.neuron_mode=='legacy' else cfg.content_T
                    du=2*c*(1-c.square())/ct/cfg.N*strength.repeat_interleave(cfg.N,1)
                    self._field_update(du,cache)
                    dIp+=cache['input_coeff']*du
                    l=F.softplus(cfg.ff_theta-g) if sgn>0 else F.softplus(g-cfg.ff_theta)
                    ff_loss+=float((l*active[:,None]).sum())*cfg.ff_weight/(int(active.sum())*cfg.L*H)
                self._input_update(dIp,input_byte)
        # Optional damping uses neuronal activity, NOT squared prediction error.
        # It changes the objective/rule and remains off in exact-gradient tests.
        if cfg.oja:
            rate=out['msgs'][-1][active].square().mean(0)
            pair=.5*(rate[:,None]+rate[None,:])
            self.grad['S']-=cfg.oja*pair*self.S
            self.grad['A']-=cfg.oja*pair*self.A
        self.grad_ticks+=1;self.seen_targets+=int(nv)
        # Full all-horizon utility, reported for EVERY transition, not used as
        # a positive scalar multiplication falsely advertised as reinforcement.
        C=torch.stack(per_hop,1);cost=torch.stack(costs,1)
        rewards=C[:,:-1]-C[:,1:]-cost[:,1:]
        return {'valid_targets':int(nv),'local_loss':loss_sum,'ff_loss':ff_loss,
                'R':float(rewards[active].mean()) if H>1 else 0.,
                'reward_by_hop':rewards.detach(), 'cost_by_hop':cost.detach()}

    def _blocks(self,name: str):
        """Разбиение параметра на блоки по уровням: у каждого уровня свой темп."""
        cfg=self.cfg
        if name in ('E','E_bias'):return [(l,) for l in range(cfg.L)]
        if name in ('S','A'):return [(slice(None),slice(l*cfg.N,(l+1)*cfg.N)) for l in range(cfg.L)]
        if name=='gate':return [(slice(l*cfg.N,(l+1)*cfg.N),) for l in range(cfg.L)]
        return [()]

    @torch.no_grad()
    def _bb_scale(self,key,p_blk,g_blk,d_blk,applied: float) -> float:
        """Темп блока из его же двух последних шагов (Барзилай–Борвейн). Ничего, кроме собственных
        величин блока, не используется: ни отложенных данных, ни дополнительных прогонов."""
        st=self.bb.setdefault(key,{'p':None,'g':None,'s':1.})
        if st['p'] is not None:
            dp=p_blk-st['p'];dg=g_blk-st['g']
            den=float((dg*dg).sum())
            num=float((dp*dg).sum())
            if den>1e-30:
                if num<=0:st['s']*=.7                      # вдоль шага не выпукло — убавить
                else:
                    eta=num/den                             # обратная кривизна вдоль пройденного
                    want=eta*float(g_blk.norm())/max(float(d_blk.norm())*applied,1e-30)
                    st['s']=(1-self.cfg.bb_ema)*st['s']+self.cfg.bb_ema*want
                st['s']=min(max(st['s'],self.cfg.bb_lo),self.cfg.bb_hi)
        st['p']=p_blk.clone();st['g']=g_blk.clone()
        return st['s']

    @torch.no_grad()
    def apply_batch(self):
        if self.grad_ticks==0:return {'no_update':True}
        info={};ticks=self.grad_ticks
        for name in self.param_names:
            frozen=('W' if name in ('S','A') else 'E' if name=='E_bias' else name)
            if frozen in self.cfg.freeze:continue
            if name=='E_in' and self.cfg.tie_input:continue
            if name=='phi' and not self.cfg.use_phase:continue
            sym=1 if name=='S' or (name=='gate' and self.cfg.symmetric_gate) else -1 if name=='A' else 0
            mask=self.mask if name in ('S','A','gate') else None
            g=project(self.grad[name]/ticks,sym,mask)
            if not bool(torch.isfinite(g).all()):raise FloatingPointError(f'nonfinite signal in {name}')
            # The classifier/embedding uses row updates, not Muon by default.
            d,conf=self.rates[name].direction(g,sym=sym,mask=mask,
                muon=self.cfg.muon and name in ('S','A'),iters=self.cfg.muon_iters)
            scale=self.cfg.head_scale if name in ('E','E_bias','E_in') else self.cfg.gate_scale if name=='gate' else self.cfg.phase_scale if name=='phi' else 1.
            p=getattr(self,name)
            if self.cfg.optimizer=='adam':
                st=self.adam.setdefault(name,{'m':torch.zeros_like(p),'v':torch.zeros_like(p),'t':0})
                st['t']+=1;b1,b2=self.cfg.adam_betas
                st['m'].mul_(b1).add_(g,alpha=1-b1)
                st['v'].mul_(b2).addcmul_(g,g,value=1-b2)
                mh=st['m']/(1-b1**st['t']);vh=st['v']/(1-b2**st['t'])
                step=mh/(vh.sqrt()+self.cfg.adam_eps)
                p.add_(project(step,sym,mask) if (sym or mask is not None) else step,
                       alpha=self.zeta*scale)
                info[f'signal_{name}']=float(g.norm());info[f'conf_{name}']=conf
                info[f'step_{name}']=float(step.norm())*self.zeta*scale
                continue
            if self.cfg.bb:
                base=self.zeta*scale
                for bi,blk in enumerate(self._blocks(name)):
                    s=self._bb_scale((name,bi),p[blk],g[blk],d[blk],base)
                    p[blk].add_(d[blk],alpha=base*s)
                    if bi==0:info[f'bb_{name}']=s
            else:
                p.add_(d,alpha=self.zeta*scale)
            if not bool(torch.isfinite(p).all()):raise FloatingPointError(f'nonfinite parameter in {name}; lower the optimization step')
            info[f'signal_{name}']=float(g.norm());info[f'conf_{name}']=conf;info[f'step_{name}']=float(d.norm())*self.zeta*scale
        self.gate.clamp_(0,1);self.gate*=self.mask
        self.phi.remainder_(2*math.pi)
        if self.act_count and 'theta' not in self.cfg.freeze:
            self.theta+=self.cfg.homeo_rate*(self.act_sum/self.act_count-self.cfg.target_act)
        for g in self.grad.values():g.zero_()
        self.grad_ticks=0;self.act_sum.zero_();self.act_count=0;self.updates+=1
        info.update({'S_norm':float(self.S.norm()),'A_norm':float(self.A.norm()),'zeta':self.zeta})
        return info

    @torch.no_grad()
    def adapt_pace(self,before: float,after: float):
        """Optional *development-set* adaptation, not evidence of self-stability."""
        if not self.cfg.pace:return self.zeta
        if not (math.isfinite(before) and math.isfinite(after)):raise FloatingPointError('nonfinite development loss')
        self.zeta*=math.exp(self.cfg.pace_strength*math.tanh((before-after)/.01))
        self.zeta=min(max(self.zeta,1e-8),1.)
        return self.zeta

    def checkpoint(self):
        if self.grad_ticks or self.act_count:raise RuntimeError('save at batch boundaries, or serialize live data iterator/state')
        return {'version':1,'cfg':asdict(self.cfg),'params':{n:getattr(self,n).clone() for n in (*self.param_names,'theta')},
                'rates':{n:r.state_dict() for n,r in self.rates.items()},'zeta':self.zeta,
                'updates':self.updates,'seen_targets':self.seen_targets}

    @classmethod
    def from_checkpoint(cls,d: dict,device: str|None=None):
        c=dict(d['cfg'])
        if device is not None:c['device']=device
        m=cls(Cfg(**c))
        for n,p in d['params'].items():getattr(m,n).copy_(p.to(m.dev))
        for n,s in d['rates'].items():m.rates[n].load_state_dict({k:(v.to(m.dev) if isinstance(v,torch.Tensor) else v) for k,v in s.items()})
        m.zeta=float(d['zeta']);m.updates=int(d['updates']);m.seen_targets=int(d['seen_targets'])
        return m


def targets(x: torch.Tensor,t: int,H: int,P: int,end: torch.Tensor):
    idx=(t+torch.arange(1,H+1,device=x.device))[None].expand(x.shape[0],-1)
    V=(idx>=P)&(idx<end[:,None])&(idx<x.shape[1])
    return x.gather(1,idx.clamp(0,x.shape[1]-1)),V


def doc_end(b):return b.P+b.loss_mask[:,b.P-1:].sum(1).long()


@torch.no_grad()
def run_prompt(mach: RREM,b):
    st=mach.init_state(b.x.shape[0])
    for t in range(b.P-1):
        act=b.active[:,t]
        if bool(act.any()):mach.advance(st,mach.tick(st,mach.input_drive(b.x[:,t])),act)
    return st


@torch.no_grad()
def evaluate(mach: RREM,batches: Iterable,*,hops: int|None=None):
    cfg=mach.cfg;H=cfg.hops if hops is None else hops
    ce=torch.zeros(cfg.L,cfg.H_pred,device=mach.dev,dtype=torch.float64);cnt=torch.zeros_like(ce)
    hc=torch.zeros(H,cfg.H_pred,device=mach.dev,dtype=torch.float64);hn=torch.zeros_like(hc)
    correct=torch.zeros_like(ce)
    for b in batches:
        b=b.to(mach.dev);end=doc_end(b)
        # Whole-history budget uses the same requested hops, not full-budget warmup.
        st=mach.init_state(b.x.shape[0])
        for t in range(b.T-1):
            act=b.active[:,t]
            if not bool(act.any()):continue
            out=mach.tick(st,mach.input_drive(b.x[:,t]),hops=H)
            if t>=b.P-1:
                Y,V=targets(b.x,t,cfg.H_pred,b.P,end);V=V&act[:,None]
                for l in range(cfg.L):
                    lg=mach.logits(out['msgs'][-1],l)
                    c=-lg.log_softmax(-1).gather(-1,Y[:,:,None]).squeeze(-1)
                    ce[l]+=(c*V).double().sum(0);cnt[l]+=V.sum(0)
                    correct[l]+=((lg.argmax(-1)==Y)&V).sum(0)
                for k,msg in enumerate(out['msgs']):
                    lp=mach.logits(msg,0).log_softmax(-1)
                    h=-lp.gather(-1,Y[:,:,None]).squeeze(-1)
                    hc[k]+=(h*V).double().sum(0);hn[k]+=V.sum(0)
            mach.advance(st,out,act,plastic=False)
    nan=torch.full_like(ce,float('nan'));bpb=torch.where(cnt>0,ce/cnt.clamp_min(1)/math.log(2),nan)
    hop=torch.where(hn>0,hc/hn.clamp_min(1)/math.log(2),torch.full_like(hc,float('nan')))
    rl=mach.read_levels[-1]  # заголовочные числа берём со слоя, на котором висит словарь
    return {'bpb':bpb.cpu().tolist(),'read_level':rl,'bpb_h1':float(bpb[rl,0]),'bpb_mean_all_h':float(bpb[rl].mean()),
            'counts':cnt.long().cpu().tolist(),'accuracy':(correct/cnt.clamp_min(1)).cpu().tolist(),
            'hop_curve':hop[:,0].cpu().tolist(),'hop_curve_all_h':hop.cpu().tolist(),'whole_history_hops':H}


@torch.no_grad()
def train_batch(mach: RREM,b,*,generator: torch.Generator|None=None):
    b=b.to(mach.dev);st=run_prompt(mach,b);end=doc_end(b);stats=[]
    if generator is None:generator=torch.Generator().manual_seed(mach.cfg.seed+mach.updates)
    for t in range(b.P-1,b.T-1):
        act=b.active[:,t]
        if not bool(act.any()):continue
        byte=b.x[:,t];out=mach.tick(st,mach.input_drive(byte),learn=True)
        Y,V=targets(b.x,t,mach.cfg.H_pred,b.P,end);V&=act[:,None]
        negative=None
        if mach.cfg.ff_weight and st.p_prev is not None and (t-(b.P-1))%mach.cfg.ff_every==0 and bool(V.any()):
            nb=torch.multinomial(st.p_prev.cpu(),1,generator=generator).squeeze(1).to(mach.dev)
            negative=(mach.tick(st.detach_clone(),mach.input_drive(nb),learn=True),nb)
        stats.append(mach.learn_tick(st,out,byte,Y,V,negative=negative))
        p=mach.logits(out['msgs'][-1],0)[:,0].softmax(-1)
        st.p_prev=p if st.p_prev is None else torch.where(act[:,None],p,st.p_prev)
        # The last context byte predicts the first response byte and may learn.
        mach.advance(st,out,act,plastic=False)
        supervised=V.any(-1)
        if bool(supervised.any()):
            mach.act_sum+=out['msgs'][-1][supervised].abs().sum(0)
            mach.act_count+=int(supervised.sum())
    info=mach.apply_batch()
    good=[s for s in stats if s['valid_targets']]
    info['local_loss']=sum(s['local_loss'] for s in good)/max(len(good),1)
    info['R']=sum(s['R'] for s in good)/max(len(good),1)
    return info


@torch.no_grad()
def generate_bytes(mach: RREM, prompt: str | bytes, n_bytes: int, *, temperature: float=1., seed: int=0) -> bytes:
    """Ordinary causal byte generation; all eight horizons are computed, h1 is sampled.

    This intentionally does not falsely interpret independent horizon marginals
    as a coherent joint distribution of eight bytes. All shared parameters and
    homeostatic statistics remain unchanged. An empty prompt is not inferred.
    """
    if n_bytes<0 or temperature<0:raise ValueError('n_bytes and temperature must be nonnegative')
    raw=prompt.encode('utf-8') if isinstance(prompt,str) else bytes(prompt)
    if not raw:raise ValueError('supply a nonempty prompt')
    st=mach.init_state(1);active=torch.ones(1,dtype=torch.bool,device=mach.dev)
    for b in raw[:-1]:
        x=torch.tensor([b],device=mach.dev,dtype=torch.long)
        out=mach.tick(st,mach.input_drive(x));mach.advance(st,out,active)
    current=raw[-1];result=[];gen=torch.Generator().manual_seed(seed)
    for _ in range(n_bytes):
        x=torch.tensor([current],device=mach.dev,dtype=torch.long)
        out=mach.tick(st,mach.input_drive(x));logits=mach.logits(out['msgs'][-1],0)[0,0]
        if temperature==0:current=int(logits.argmax())
        else:current=int(torch.multinomial((logits/temperature).softmax(-1).cpu(),1,generator=gen))
        result.append(current);mach.advance(st,out,active)
    return bytes(result)


def selfcheck(data,cfg: Cfg):
    """Validate ALL rows, valid offsets, document ends and literal data alignment."""
    batches=data.heldout_batches(2,8,seed=2)
    checked=0
    for b in batches:
        end=doc_end(b)
        if bool((end>b.T).any()):raise AssertionError('document end beyond storage')
        for i in range(b.x.shape[0]):
            idx=torch.nonzero(b.active[i,:b.P]).flatten()
            if idx.numel() and hasattr(data,'prompts') and hasattr(b,'doc_ids'):
                start=int(idx[0]);raw=bytes(b.x[i,start:b.P].tolist());di=int(b.doc_ids[i])
                assert raw==data.prompts[di][-len(raw):],('prompt mismatch',di)
                response=bytes(b.x[i,b.P:int(end[i])].tolist())
                assert data.responses[di].startswith(response),('response mismatch',di)
        for t in range(b.P-1,b.T):
            Y,V=targets(b.x,t,cfg.H_pred,b.P,end)
            for i in range(b.x.shape[0]):
                for h in range(cfg.H_pred):
                    pos=t+h+1;expected=b.P<=pos<min(int(end[i]),b.T)
                    assert bool(V[i,h])==expected
                    if expected:assert int(Y[i,h])==int(b.x[i,pos])
                    checked+=1
    print(f'data selfcheck: {checked} horizon/boundary assertions passed')


@dataclass
class ByteBatch:
    x: torch.Tensor
    active: torch.Tensor
    loss_mask: torch.Tensor
    P: int
    doc_ids: torch.Tensor
    @property
    def T(self):return self.x.shape[1]
    def to(self,device):
        return ByteBatch(self.x.to(device),self.active.to(device),self.loss_mask.to(device),self.P,self.doc_ids.to(device))


class JsonlData:
    """Explicit OpenOrca exports only. Splits are supplied, never guessed.

    No artificial examples are generated. Original row IDs are checked across
    train/development/test. Real-response truncation is reported by CLI settings.
    """
    def __init__(self,train_file: Path,dev_file: Path,test_file: Path|None,resp_max=64,prompt_max=256):
        if resp_max<1 or prompt_max<1:raise ValueError('prompt_max and resp_max must be positive')
        self.rows=[];self.splits={};self.resp_max=resp_max;self.prompt_max=prompt_max
        seen=set()
        for name,path in (('train',train_file),('dev',dev_file),('test',test_file)):
            ids=[]
            if path is not None:
                for line in path.read_text(encoding='utf-8').splitlines():
                    if not line.strip():continue
                    r=json.loads(line)
                    key=str(r['id'])
                    if key in seen:raise ValueError(f'duplicate/overlapping document ID: {key}')
                    seen.add(key)
                    prompt=r.get('prompt')
                    if prompt is None:prompt=str(r.get('system_prompt',''))+'\n'+str(r['question'])+'\n'
                    response=r['response']
                    if not isinstance(prompt,str) or not isinstance(response,str):raise ValueError('text fields must be strings')
                    if not prompt:prompt='\n'  # interface separator, not a synthetic training example
                    if not response:continue
                    ids.append(len(self.rows));self.rows.append((prompt.encode('utf-8'),response.encode('utf-8')))
            self.splits[name]=ids
        if not self.splits['train'] or not self.splits['dev']:raise ValueError('nonempty train and dev are required')
        self.prompts=[r[0] for r in self.rows];self.responses=[r[1] for r in self.rows]

    def pack(self,ids):
        if not ids:raise ValueError('cannot pack an empty batch')
        prompts=[self.prompts[i][-self.prompt_max:] for i in ids]
        responses=[self.responses[i][:self.resp_max] for i in ids]
        P=max(map(len,prompts));R=max(map(len,responses));B=len(ids);T=P+R
        x=torch.zeros(B,T,dtype=torch.long);active=torch.zeros_like(x,dtype=torch.bool);mask=torch.zeros_like(x,dtype=torch.bool)
        for i,(p,r) in enumerate(zip(prompts,responses)):
            start=P-len(p);end=P+len(r)
            x[i,start:P]=torch.tensor(list(p));x[i,P:end]=torch.tensor(list(r));active[i,start:end]=True
            mask[i,P-1:end-1]=True
        return ByteBatch(x,active,mask,P,torch.tensor(ids))

    def train_batches(self,seed,batch):
        gen=random.Random(seed);ids=self.splits['train'][:]
        while True:
            gen.shuffle(ids)
            for i in range(0,len(ids),batch):yield self.pack(ids[i:i+batch])

    def heldout_batches(self,n,batch,seed=2):
        ids=self.splits['dev'][:];random.Random(seed).shuffle(ids)
        return [self.pack(ids[i:i+batch]) for i in range(0,min(len(ids),n*batch),batch)]

    def test_batches(self,batch):
        ids=self.splits['test']
        return [self.pack(ids[i:i+batch]) for i in range(0,len(ids),batch)]


def override_cfg(cfg: Cfg,items: list[str]):
    known={f.name for f in fields(cfg)}
    for item in items:
        name,value=item.split('=',1)
        if name not in known:raise ValueError(f'unknown setting: {name}')
        old=getattr(cfg,name)
        if isinstance(old,bool):
            if value.lower() not in ('true','false','1','0','да','нет'):raise ValueError(f'invalid boolean: {item}')
            new=value.lower() in ('true','1','да')
        elif isinstance(old,tuple):
            cast=float if name=='trace_taus' else int if name=='delay_lags' else str
            new=tuple(cast(x) for x in value.split(',') if x)
        else:new=type(old)(value)
        setattr(cfg,name,new)
    cfg.validate();return cfg


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--N',type=int,default=256);ap.add_argument('--L',type=int,default=2)
    ap.add_argument('--hops',type=int,default=8);ap.add_argument('--steps',type=int,default=60)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--resp-max',type=int,default=64)
    ap.add_argument('--prompt-max',type=int,default=256);ap.add_argument('--eval-every',type=int,default=10)
    ap.add_argument('--eval-batches',type=int,default=3);ap.add_argument('--out',type=Path,default=Path('runs/rrem_repaired'))
    ap.add_argument('--name',default='local');ap.add_argument('--micro',action='store_true')
    ap.add_argument('--selfcheck',action='store_true');ap.add_argument('--resume',type=Path)
    ap.add_argument('--train-jsonl',type=Path);ap.add_argument('--dev-jsonl',type=Path);ap.add_argument('--test-jsonl',type=Path)
    ap.add_argument('--set',nargs='*',default=[])
    args=ap.parse_args()
    if min(args.batch,args.eval_every,args.eval_batches,args.resp_max,args.prompt_max)<1:ap.error('batch and evaluation/data sizes must be positive')
    if args.steps<0:ap.error('steps must be nonnegative')
    if (args.dev_jsonl or args.test_jsonl) and not args.train_jsonl:ap.error('explicit dev/test JSONL requires --train-jsonl')
    if args.micro:
        args.N=256;args.L=2;args.hops=8;args.resp_max=64
    cfg=override_cfg(Cfg(N=args.N,L=args.L,hops=args.hops),args.set)
    if args.train_jsonl:
        if not args.dev_jsonl:ap.error('--dev-jsonl required with --train-jsonl')
        data=JsonlData(args.train_jsonl,args.dev_jsonl,args.test_jsonl,args.resp_max,args.prompt_max)
    else:
        try:
            from drrem.config import DataConfig
            from drrem.data.openorca import OpenOrcaBytes
        except ImportError as e:
            raise SystemExit('Original OpenOrca loader is not installed. Run within your drrem project or supply explicit OpenOrca JSONL exports.') from e
        data=OpenOrcaBytes(DataConfig(resp_max=args.resp_max,batch=args.batch))
    selfcheck(data,cfg)
    if args.selfcheck:return
    generator=torch.Generator().manual_seed(cfg.seed+11);start=0
    if args.resume:
        ck=torch.load(args.resume,map_location=cfg.device,weights_only=True)
        saved=dict(ck['machine']['cfg']);saved['device']=cfg.device
        if saved!=asdict(cfg):raise ValueError('resume config differs; repeat the original --set/settings')
        mach=RREM.from_checkpoint(ck['machine'],cfg.device);start=mach.updates
        generator.set_state(ck['generator'])
    else:mach=RREM(cfg)
    it=data.train_batches(cfg.seed+3,args.batch)
    for _ in range(start):next(it)  # deterministic batch-boundary replay of original sampler
    dev=data.heldout_batches(args.eval_batches,args.batch,seed=2)
    dev_ids=set(int(i) for b in dev for i in b.doc_ids.tolist())
    args.out.mkdir(parents=True,exist_ok=True)
    log=args.out/(args.name+'.jsonl');ckpath=args.out/(args.name+'.pt')
    if log.exists() and not args.resume:raise FileExistsError(f'{log} already exists; use a new --name or --resume')
    if start>args.steps:raise ValueError('requested total steps precede the resumed checkpoint')
    def write(rec):
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print(json.dumps(rec,ensure_ascii=False),flush=True)
    write({'step':start,'kind':'development_not_test','eval':evaluate(mach,dev)})
    for step in range(start+1,args.steps+1):
        b=next(it)
        if any(int(i) in dev_ids for i in b.doc_ids.tolist()):raise ValueError('training/development ID overlap')
        before=evaluate(mach,dev)['bpb_mean_all_h'] if cfg.pace else float('nan')
        info=train_batch(mach,b,generator=generator)
        if cfg.pace:
            after=evaluate(mach,dev)['bpb_mean_all_h'];mach.adapt_pace(before,after)
            info.update({'development_before':before,'development_after':after})
        rec={'step':step,**info}
        if step%args.eval_every==0 or step==args.steps:
            rec['eval']=evaluate(mach,dev)
            torch.save({'machine':mach.checkpoint(),'generator':generator.get_state()},ckpath)
        write(rec)
    if isinstance(data,JsonlData) and data.splits['test']:
        write({'kind':'final_test_once','step':mach.updates,'eval':evaluate(mach,data.test_batches(args.batch))})


if __name__=='__main__':main()
