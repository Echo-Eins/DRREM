import torch

from drrem.core.byte_prior import count_documents,SparseBytePrior


def test_counts_have_response_mask_and_never_cross_documents():
    table=count_documents([(b'abcX',3),(b'abcY',3),(b'qqqZ',4)],(1,3))
    assert table['response_count']==2
    expected={int.from_bytes(b'abcX','big'):1,int.from_bytes(b'abcY','big'):1}
    assert dict(zip(table['3']['keys'].tolist(),table['3']['counts'].tolist()))==expected
    table=count_documents([(b'a',0),(b'b',0),(b'c',0)],(1,))
    assert len(table['1']['keys'])==0


def test_prior_alignment_causality_padding_and_unseen_backoff():
    table=count_documents([(b'abcX'*8,3),(b'abcY',3)],(1,3))
    m=SparseBytePrior(table)
    ids=torch.tensor([list(b'abcabc')]);p=m(ids)
    assert p[0,2,ord('X')]>p[0,2,ord('Y')]>p[0,2,ord('Z')]
    changed=ids.clone();changed[:,3:]=ord('z')
    torch.testing.assert_close(m(changed)[:,:3],p[:,:3],rtol=0,atol=0)
    torch.testing.assert_close(p.sum(-1),torch.ones_like(ids,dtype=torch.float32))
    short=m(torch.tensor([[ord('c')]]));padded=m(torch.tensor([[0,0,ord('c')]]),torch.tensor([[False,False,True]]))
    torch.testing.assert_close(short[:,0],padded[:,-1],rtol=0,atol=0)
    legacy=SparseBytePrior(table,style='fullcascade')(ids)
    assert torch.isfinite(legacy).all() and legacy.min()>0
