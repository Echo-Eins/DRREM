"""Fresh generated confirmation of the prespecified final curriculum endpoint."""
from pathlib import Path
import json
import numpy as np
import torch
from scripts.audit_transport_functions import load, save
from scripts.probe_binding_learnability import example,TEST_PEOPLE,evaluate
from scripts.probe_binding_limits import generate_score


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    rng=np.random.default_rng(220922819);suites={}
    for family in ['shared_digits','shared_letters','unrestricted_digits']:
        rows=[]
        for i in range(256):
            task=example(rng,TEST_PEOPLE,['question','demonstration'][i%2],True,True)
            alphabet=list('0123456789' if 'digits' in family else 'bcdfghjkmnpqrstvwxyz')
            if family.startswith('shared'):
                prefix=''.join(rng.choice(alphabet,size=2));codes=[prefix+c for c in rng.choice(alphabet,size=4,replace=False)]
            else:
                codes=[]
                while len(codes)<4:
                    c=''.join(rng.choice(alphabet,size=3))
                    if c not in codes:codes.append(c)
            raw=task['prefix']
            for k,old in enumerate(task['codes']):raw=raw.replace(old.encode(),f'<VALUE{k}>'.encode())
            for k,new in enumerate(codes):raw=raw.replace(f'<VALUE{k}>'.encode(),new.encode())
            rows.append({**task,'prefix':raw,'codes':codes})
        suites[family]=rows
    root=Path('runs/functional_map_20260922');result=dict(scope='new seed,768 new tasks,random per-task value prefixes; diagnostic recall, not independent natural-language test',models={})
    for name,path in [('language_control',root/'curriculum/0.0.pt'),('curriculum',root/'curriculum/0.1.pt')]:
        m,_=load(path);row={}
        for family,cases in suites.items():
            row[family]=dict(choice=evaluate(m,cases),generation=generate_score(m,cases))
            print(json.dumps(dict(model=name,family=family,**row[family])),flush=True)
            result['models'][name]=row;save(root/'curriculum_confirmation.json',result)
        del m;torch.cuda.empty_cache()


if __name__=='__main__':main()
