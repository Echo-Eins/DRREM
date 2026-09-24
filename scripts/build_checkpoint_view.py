"""Export a real checkpoint into an offline viewer using its frozen source tree.

Called by build_transport_view.py --checkpoint ... --source-root ... .
CPU FP32 diagnostic replay, not a recording of a historical CUDA training step.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**20),b''): h.update(part)
    return h.hexdigest()


def helper(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--text',default='Nim=417; Nim=')
    p.add_argument('--text-file',type=Path)
    p.add_argument('--route-seed',type=int,default=0)
    p.add_argument('--label')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1];source=a.source_root.resolve()
    sys.path.insert(0,str(source))
    import torch
    torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    protocol=ck['protocol']
    if not protocol.get('source_hashes'):raise ValueError('Checkpoint has no source provenance')
    def verify_sources():
        for file, expected in protocol['source_hashes'].items():
            if digest(source/file)!=expected:raise ValueError('Frozen source mismatch: '+file)
    verify_sources()
    packet='paths' in protocol['model']
    if packet:
        from drrem.core.packet_transport import PacketTransportMachine,PacketTransportConfig
        model=PacketTransportMachine(PacketTransportConfig(**protocol['model']))
    else:
        if protocol.get('factory')!='scripts.train_full_signal_trial.make_trial_model':
            raise ValueError('Unsupported checkpoint factory: refusing to guess its forward')
        from drrem.core.causal_transport import CausalTransportConfig
        from scripts.train_full_signal_trial import make_trial_model
        model=make_trial_model(protocol['variant'],CausalTransportConfig(**protocol['model']))
    if not Path(inspect.getfile(type(model))).resolve().is_relative_to(source):
        raise ValueError('Loaded model is not from the declared source tree')
    model.load_state_dict(ck['model'],strict=True);model.eval()
    raw=a.text_file.read_bytes() if a.text_file else a.text.encode('utf-8')
    if not 1<=len(raw)<=127:raise ValueError('Supply 1..127 raw prefix bytes; no silent truncation')
    ids=torch.tensor([[256,*raw]])
    dense=helper('checkpoint_dense_recorder',root/'drrem/diagnostics/transport_view.py')
    if packet:
        observer=helper('checkpoint_packet_recorder',root/'drrem/diagnostics/packet_view.py')
        trace,audit=observer.record_packet(model,ids,a.route_seed)
        parameters={}
    else:
        trace,audit=dense.record_forward(model,ids)
        parameters=dense.export_parameters(model)
    assert all(torch.equal(v,ck['model'][k]) for k,v in model.state_dict().items())
    assert all(q.grad is None for q in model.parameters())
    verify_sources()
    provenance=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest(a.checkpoint),
        source_root=str(source),source_hashes_verified=len(protocol['source_hashes']),
        step=ck['step'],supervised_bytes=ck['raw_byte_exposures'],
        context_byte_exposures=ck.get('context_byte_exposures'),
        initialization=protocol.get('initialization'),parent=protocol.get('parent'),
        configuration=protocol['model'],variant=protocol.get('variant','packet'),
        optimizer=protocol.get('optimizer'),training_compile_hops=protocol.get('compile_hops',False),
        execution='CPU FP32 eager diagnostic replay; NOT CUDA BF16/compiled training replay',
        runtime_model_training=False,weights_unchanged=True,gradients_created=False,
        route_seed=a.route_seed if packet else None,
        prefix_raw_hex=raw.hex(),prefix_sha256=hashlib.sha256(raw).hexdigest(),
        prefix_boundary='Fresh document BOS, no omitted earlier context',
        recorder_sha256={str(f.relative_to(root)):digest(f) for f in [
            root/'drrem/diagnostics/packet_view.py',root/'drrem/diagnostics/transport_view.py',
            Path(__file__),root/'scripts/checkpoint_view.html']})
    statistics={}
    if packet:
        chosen=trace['routes'].long();previous=trace['previous'].long();n=model.cfg.neurons
        bridges=((chosen//n-previous//n).abs()==2);bridges[0]=False
        statistics=dict(executed_visits=chosen.numel(),distinct_nodes=int(chosen.unique().numel()),
            bridge_visits=int(bridges.sum()),mean_route_kl_uniform=float(trace['kl_uniform'].mean()),
            mean_attention_increment_rms=float((model.step_scale*trace['attention']).square().mean(-1).sqrt().mean()),
            mean_mlp_increment_rms=float((model.step_scale*trace['mlp']).square().mean(-1).sqrt().mean()))
    else:
        statistics={key:float((trace[key]*(1 if key=='bridge' else model.step_scale)).square().mean().sqrt())
                    for key in ['spatial','attention','mlp','bridge']}
    payload=dict(version=2,kind='packet' if packet else 'dense',cfg=asdict(model.cfg),
        label=a.label or a.checkpoint.parent.name,step_scale=model.step_scale,
        ids=ids[0].tolist(),text=raw.decode('utf-8',errors='replace'),
        trace={k:dense.packed(v) for k,v in trace.items()},parameters=parameters,
        provenance=provenance,audit=audit,statistics=statistics,
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    template=(root/'scripts/checkpoint_view.html').read_text()
    assert template.count('__TRANSPORT_DATA__')==1
    for relative,expected in provenance['recorder_sha256'].items():
        if digest(root/relative)!=expected:raise ValueError('Recorder changed during export: '+relative)
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(template.replace('__TRANSPORT_DATA__',data))
    report=dict(provenance=provenance,audit=audit,statistics=statistics,html_bytes=a.out.stat().st_size,
                html_sha256=digest(a.out))
    a.out.with_suffix('.audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(out=str(a.out),audit=audit,statistics=statistics,html_bytes=report['html_bytes']),ensure_ascii=False))


if __name__=='__main__':main()
