import json,torch
from pathlib import Path
from drrem.data.protocol import file_digest,restore_openorca_protocol

torch.set_num_threads(2)
base=Path('reports/last_decoder_20260920');out=base/'scale1024x3'
meta=json.loads((out/'protocol.json').read_text())
assert all(file_digest(f)==h for f,h in meta['source_hashes'].items())
ck=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
p=ck['machine']['parameters'];c=ck['machine']['config'];N,L=c['N'],c['L']
assert p['E'].shape==(8,256,1024) and int(torch.count_nonzero(p['E'][0]))==256*1024
assert p['S'].shape==p['A'].shape==(4,3072,3072)
level=torch.arange(N*L)//N;mask=(level[:,None]-level[None,:]).abs()<=1;mask.fill_diagonal_(False)
zero_edges=[]
for k in range(4):
 assert torch.equal(p['S'][k],p['S'][k].T) and torch.equal(p['A'][k],-p['A'][k].T)
 W=p['S'][k]+p['A'][k]
 assert torch.count_nonzero(W[~mask])==0
 for i,j in torch.nonzero((W==0)&mask).tolist():
  # Accidental cancellation at initialization is not a missing parameter/edge.
  assert p['S'][k,i,j]!=0 and p['A'][k,i,j]!=0
  zero_edges.append({'delay_index':k,'post':i,'pre':j,'S':float(p['S'][k,i,j]),'A':float(p['A'][k,i,j])})
assert len(zero_edges)<10
saved_updates=ck['machine']['updates'];del ck,p,W
D=restore_openorca_protocol(meta['protocol']);plan=meta['protocol']['response_budget'];order=plan['order'];B=64
lengths=[max(min(len(D.prompts[i]),1024) for i in order[j:j+B])+max(len(D.responses[i]) for i in order[j:j+B])-1 for j in range(0,len(order),B)]
rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()];updates=[r for r in rows if 'info' in r]
seconds_per_position=sum(r['info']['seconds'] for r in updates)/sum(lengths[:len(updates)]) if updates else None
result={'shape_checks':'passed on actual large checkpoint','encoder_shape':[256,1024],
 'readout_shape':[8,256,1024],'S_A_shape':[4,3072,3072],'permitted_edges_per_delay':7*N*N-3*N,
 'accidental_zero_S_plus_A_on_trainable_edges':zero_edges,
 'trainable_parameter_count':2*4*(N*L)**2+8*256*N+8*256,
 'source_hashes_match':True,'planned_batches':len(lengths),'planned_padded_byte_positions':sum(lengths),
 'checkpoint_updates':saved_updates,'completed_batches':len(updates),
 'seen_response_bytes':updates[-1]['seen_response_bytes'] if updates else 0,
 'estimated_training_hours_excluding_eval':sum(lengths)*seconds_per_position/3600 if seconds_per_position else None}
(base/'large_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
