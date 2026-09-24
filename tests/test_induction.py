import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from drrem.core.induction import induction_candidates,suffix_candidates


def test_exact_fullcascade_induction_forward_contract():
    path=Path('/home/echoens/Coding/Python/Mythos_P/training/full_cascade.py')
    tree=ast.parse(path.read_text());node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_induction_q')
    node.decorator_list=[];space={'torch':torch,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),space)
    torch.manual_seed(71);features=torch.randn(2,30,16);ids=torch.randint(8,(2,30))
    cfg=SimpleNamespace(ind_near=3,ind_topm=8,ind_beta=6.,ind_span=20,vocab=8)
    ref,cos,live=space['_induction_q'](cfg,features,ids)
    out,stats,_=induction_candidates(features,ids,near=3,span=20,vocab=8)
    torch.testing.assert_close(ref,out,rtol=0,atol=0)
    torch.testing.assert_close(stats[...,-1].bool(),live)
    torch.testing.assert_close(stats[...,0][live],cos[live],rtol=0,atol=0)


def test_continuation_alignment_future_invariance_and_live_query_gradient():
    torch.manual_seed(19);features=torch.randn(2,18,12,requires_grad=True);ids=torch.randint(8,(2,18))
    q,stats,trace=induction_candidates(features,ids,near=2,topm=4,vocab=8)
    assert ((trace['indices'][:,8:]+1)<torch.arange(8,18)[None,:,None]).all()
    changed=ids.clone();changed[:,13:]=(changed[:,13:]+1)%8
    altered=features.detach().clone();altered[:,13:]+=5
    other,_,_=induction_candidates(altered,changed,near=2,topm=4,vocab=8)
    torch.testing.assert_close(q[:,:13],other[:,:13],rtol=0,atol=0)
    q[:,12,3].sum().backward();assert features.grad[:,:13].norm()>0 and features.grad[:,13:].count_nonzero()==0
    assert torch.isfinite(stats).all()
    # Prefix growth cannot rescale the statistics at a fixed query.
    short=induction_candidates(features[:,:13],ids[:,:13],near=2,topm=4,vocab=8)
    torch.testing.assert_close(short[1][:,12],stats[:,12],rtol=2e-6,atol=2e-7)


def test_exact_suffix_reads_known_continuation_only():
    ids=torch.tensor([list(b'abcX---abc')])
    q=suffix_candidates(ids,context=3,near=0)
    assert q[0,-1,ord('X')]==1 and q[0,2].sum()==0
