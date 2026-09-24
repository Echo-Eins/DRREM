"""One locked independent evaluation after the complete 10 MB data pass."""
import json
from pathlib import Path
import torch
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model,evaluate
from scripts.summarize_fineweb import paired


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    root=Path('runs/fineweb_energy_20260922');out=root/'independent_test.json'
    if out.exists():raise FileExistsError('this test result already exists; do not silently reopen it')
    locked=json.loads((root/'test_plan.json').read_text());corpus=FineWebBytes(DEFAULT_CACHE)
    if locked['corpus_cache_hashes']!=corpus.manifest['cache_hashes']:raise ValueError('corpus identity mismatch')
    if any(digest(corpus.cache/name)!=expected for name,expected in locked['corpus_cache_hashes'].items()):
        raise ValueError('cached corpus bytes differ from the locked manifest')
    plan=corpus.plan('test',budget=10**12,block=locked['block'],context=locked['context'],max_docs=len(locked['test_documents']))
    if plan['documents']!=locked['test_documents']:raise ValueError('test selection mismatch')
    # Validate BOTH endpoints before reading test examples.
    for name in locked['models']:
        ck=torch.load(root/name/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        if ck['raw_byte_exposures']!=locked['required_raw_training_exposures']:raise ValueError('full-budget endpoint required')
        del ck
    result=dict(plan_sha256=digest(root/'test_plan.json'),scope='previously closed512 FineWeb test documents; all selected bytes; warm OpenOrca initialization',models={})
    for name in locked['models']:
        path=root/name/'checkpoint.pt';ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model']);value=evaluate(m,corpus,plan)
        result['models'][name]=dict(checkpoint_sha256=digest(path),step=ck['step'],raw_byte_exposures=ck['raw_byte_exposures'],context_byte_exposures=ck['context_byte_exposures'],evaluation=value)
        print(json.dumps(dict(arm=name,bpb=value['bpb'],bytes=value['raw_bytes'])),flush=True)
        (root/'independent_test.partial.json').write_text(json.dumps(result,indent=2)+'\n')
        del m,ck;torch.cuda.empty_cache()
    control,candidate=(result['models'][n]['evaluation']['documents'] for n in locked['models'])
    result['comparison']=paired(candidate,control);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['comparison']),flush=True)


if __name__=='__main__':main()
