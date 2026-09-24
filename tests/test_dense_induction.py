import torch

from drrem.core.induction import induction_candidates


def test_missing_top8_value_has_no_address_gradient_but_dense_support_can_learn_it():
    torch.manual_seed(719)
    feature=torch.randn(1,100,16,requires_grad=True)
    ids=torch.zeros(1,100,dtype=torch.long)
    with torch.no_grad():
        _,_,trace=induction_candidates(feature,ids,near=64,topm=8,vocab=8)
        selected=set(trace['indices'][0,99].tolist())
        missing=next(j for j in range(35) if j not in selected)
        ids[0,missing+1]=7
    hard,_,_=induction_candidates(feature,ids,near=64,topm=8,vocab=8)
    dense,_,_=induction_candidates(feature,ids,near=64,topm=100,vocab=8)
    assert hard[0,99,7]==0 and dense[0,99,7]>0
    gh=torch.autograd.grad(hard[0,99,7],feature,retain_graph=True)[0]
    gd=torch.autograd.grad(dense[0,99,7],feature)[0]
    assert gh[0,99].count_nonzero()==0 and gd[0,99].norm()>0
    changed=ids.clone();changed[:,90:]=3
    other,_,_=induction_candidates(feature,changed,near=64,topm=100,vocab=8)
    torch.testing.assert_close(other[:,:90],dense[:,:90],rtol=0,atol=0)
