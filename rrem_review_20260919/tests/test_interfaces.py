"""Interface/protocol regression tests. Not a text-training benchmark."""
import copy,importlib.util,json,pathlib,subprocess,sys,tempfile
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_base',ROOT/'tests/test_repaired.py')
t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)
r=t.r
results={}

# Last-only CE derivative with genuine local gradient, no finite-point assumption.
m=r.RREM(t.mcfg(hop_loss='last',ff_weight=0.,lam_edge=0.,lam_spike=0.))
st=t.state_fixture(m);b=torch.tensor([32,65,195]);Y=torch.randint(0,256,(3,8));V=torch.ones_like(Y,dtype=torch.bool)
with torch.no_grad():
    out=m.tick(st,m.input_drive(b),learn=True);m.learn_tick(st,out,b,Y,V)
manual={n:v.clone() for n,v in m.grad.items()}
for n in m.param_names:setattr(m,n,getattr(m,n).detach().clone().requires_grad_(True))
u=st.u.detach();W=m.W();I=m.input_drive(b)
for cache in out['records']:
    slow=torch.einsum('bmi,mji->bj',out['ch'].detach(),W[1:])
    u=(1-m.cfg.alpha)*u+m.cfg.alpha*(cache['pre'].detach()@W[0].T+slow+I)
    msg,_=m.emitter(u,cache['theta'].detach(),cache['hop'])
loss=sum(-m.logits(msg,l).log_softmax(-1).gather(-1,Y[:,:,None]).mean() for l in range(m.cfg.L))/m.cfg.L
g=torch.autograd.grad(loss,[getattr(m,n) for n in m.param_names],allow_unused=True)
errs={}
for n,a in zip(m.param_names,g):
    if a is None:a=torch.zeros_like(manual[n])
    sym=1 if n=='S' else -1 if n=='A' else 0
    errs[n]=t.assert_close(manual[n],r.project(-a,sym,m.mask if n in ('S','A','gate') else None))
results['last_hop_loss_gradient_errors']=errs

# Sampling must neither train global homeostasis nor touch optimizer state.
m=r.RREM(t.mcfg(N=4,hops=2));snapshot=t.state_checksum(m)
prompt=(ROOT/'original/README.md').read_text(encoding='utf-8')[:40]
a=r.generate_bytes(m,prompt,16,seed=3);b=r.generate_bytes(m,prompt,16,seed=3)
assert a==b and t.equal_tree(snapshot,t.state_checksum(m))
results['generation_readonly_and_seeded']=True

# Loader/CLI tests use literal supplied prose as parser fixtures; selfcheck does
# not train any model. These files are temporary and NOT delivered as OpenOrca.
text=(ROOT/'original/README.md').read_text(encoding='utf-8')
with tempfile.TemporaryDirectory() as td:
    p=pathlib.Path(td)
    for split,start in (('train',200),('dev',600)):
        rows=[{'id':f'format_fixture_{split}_{i}','prompt':text[start+i:start+i+30+i],
               'response':text[start+40+i:start+100+i]} for i in range(3)]
        (p/f'{split}.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows),encoding='utf-8')
    d=r.JsonlData(p/'train.jsonl',p/'dev.jsonl',None,resp_max=24,prompt_max=32)
    for bt in d.heldout_batches(2,2):
        end=r.doc_end(bt)
        for i in range(bt.x.shape[0]):
            raw=bytes(bt.x[i,bt.P:int(end[i])].tolist())
            assert d.responses[int(bt.doc_ids[i])].startswith(raw)
    call=subprocess.run([sys.executable,str(ROOT/'rrem_repaired.py'),'--selfcheck','--train-jsonl',str(p/'train.jsonl'),
        '--dev-jsonl',str(p/'dev.jsonl'),'--resp-max','24','--prompt-max','32','--set','device=cpu'],capture_output=True,text=True)
    assert call.returncode==0,call.stderr
    results['jsonl_cli_selfcheck_stdout']=call.stdout.strip()
    bad=r.JsonlData
    try:bad(p/'train.jsonl',p/'train.jsonl',None)
    except ValueError:results['split_overlap_rejected']=True
    else:raise AssertionError('overlap allowed')
# Transmission cost is invariant to an inverse gate/weight rescaling.
m=r.RREM(t.mcfg(lam_edge=.3));st=t.state_fixture(m);byte=torch.tensor([32,65,195])
Y=torch.randint(0,256,(3,8));V=torch.ones_like(Y,dtype=torch.bool)
with torch.no_grad():
    out=m.tick(st,m.input_drive(byte),learn=True)
    before=m.learn_tick(st,out,byte,Y,V)['cost_by_hop'].clone()
    for g in m.grad.values():g.zero_()
    m.S.mul_(100);m.A.mul_(100);m.gate.div_(100)
    out=m.tick(st,m.input_drive(byte),learn=True)
    after=m.learn_tick(st,out,byte,Y,V)['cost_by_hop']
err=t.assert_close(before,after,1e-10)
results['cost_inverse_gate_weight_rescaling_relative_error']=err
(ROOT/'results/interface_tests.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
