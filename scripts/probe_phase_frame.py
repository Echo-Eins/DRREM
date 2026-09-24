"""Conditioning of positional codes alone, with oracle queries (not an LM)."""
import argparse
import json
import math
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    D=128;P=D//2
    frequencies={'log_rope':10000.**(-torch.arange(0,D,2,dtype=torch.float64)/D),
                 'uniform_ring64':2*math.pi*torch.arange(P,dtype=torch.float64)/P}
    result={'scope':'positional-code-only oracle: query already identifies the desired position; no learned content keys or language model',
            'key_width':D,'examples_shared_between_arms':True,'arms':{}}
    for name,frequency in frequencies.items():
        rows=[]
        for T in [32,64,96,256]:
            phase=torch.arange(T,dtype=torch.float64)[:,None]*frequency
            k=torch.stack((phase.cos(),phase.sin()),-1).flatten(-2)/math.sqrt(P)
            sv=torch.linalg.svdvals(k);mass=sv.square()/sv.square().sum()
            tokens=torch.randint(16,(1024,T),generator=torch.Generator().manual_seed(508+T))
            values=torch.nn.functional.one_hot(tokens,16).double()
            memory=torch.einsum('td,btv->bdv',k,values)
            lag=torch.where(torch.arange(1024)%2==0,4,16);query=k[T-lag]
            prediction=torch.einsum('bd,bdv->bv',query,memory).argmax(-1)
            gram=k@k.T;offdiag=gram-torch.eye(T,dtype=torch.float64)
            rows.append({'length':T,'entropy_rank_of_gram':float(torch.exp(-(mass*mass.clamp_min(1e-300).log()).sum())),
                'offdiag_rms':float(offdiag.square().sum().div(T*(T-1)).sqrt()),
                'raw_linear_read_accuracy':float((prediction==tokens[torch.arange(1024),T-lag]).float().mean())})
        result['arms'][name]=rows
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
