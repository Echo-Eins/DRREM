"""Build an offline 3-D viewer: a real checkpoint or a 128-wide demonstration.

Run: python -m scripts.build_transport_view --out docs/transport_3d.html
Real weights: add --checkpoint PATH --source-root FROZEN_TRAINING_SOURCE.
The default is a controlled initialization demonstration, never a trained
1024-wide checkpoint disguised as 128 neurons. Production model code is read,
hashed and embedded; this script does not mutate that code or train anything.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.identity_bridge import IdentityBridgeTransportMachine
from drrem.core.synaptic_basis import BridgedSynapticBasisTransportMachine
from drrem.diagnostics.transport_view import export_parameters, packed, record_forward, visualization_statistics


def build_payload(width=128, text='Nim=417; Nim=', seed=230923):
    source_root = Path(__file__).resolve().parents[1]
    sources=[]
    for relative in ['drrem/core/causal_transport.py','drrem/core/identity_bridge.py',
                     'drrem/core/synaptic_basis.py','drrem/core/ridge_metric.py',
                     'drrem/core/ridge_plasticity.py','drrem/diagnostics/transport_view.py',
                     'scripts/build_transport_view.py']:
        content=(source_root/relative).read_bytes()
        sources.append(dict(path=relative, sha256=hashlib.sha256(content).hexdigest(), text=content.decode()))
    cfg = CausalTransportConfig(neurons=width, layers=3, hops=8, vocab=257, checkpoint_hops=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    base = RidgeMetricTransportMachine(cfg).eval()
    common = {name: value.clone() for name, value in base.state_dict().items()}
    ids = torch.tensor([[256, *text.encode('utf-8')]])
    if ids.shape[1] > 128:
        raise ValueError('Use a short prefix (at most 127 UTF-8 bytes) for this offline viewer.')
    variants = []
    for name, title, kind, gain, basis_std in [
        ('base', 'База · без перемычек', 'base', None, 0.),
        ('bridges_zero', 'Перемычки · начальные g=0', 'bridge', 0., 0.),
        ('bridges_open', 'Перемычки · тестовые g=1', 'bridge', 1., 0.),
        ('kan_zero', 'KAN + перемычки · начальные a=b=g=0', 'kan', 0., 0.),
        ('kan_probe', 'KAN + перемычки · ненулевой тест', 'kan', .25, .08/(width**.5)),
    ]:
        torch.manual_seed(seed)
        model = (base if kind=='base' else IdentityBridgeTransportMachine(cfg) if kind=='bridge'
                 else BridgedSynapticBasisTransportMachine(cfg)).eval()
        result = model.load_state_dict(common, strict=False)
        allowed = ('bridge_gain.', '.coefficients', '.raw_frequency')
        if result.unexpected_keys or any(not any(x in key for x in allowed) for key in result.missing_keys):
            raise AssertionError('A base parameter was not copied exactly.')
        with torch.no_grad():
            for value in getattr(model, 'bridge_gain', {}).values(): value.fill_(gain)
            if basis_std:
                for edge in model.edges.values(): edge.coefficients.normal_(std=basis_std)
                # Explicit, labelled test activation of an otherwise zero output branch.
                model.plastic_gain.fill_(.1)
        trace, audit = record_forward(model, ids)
        variants.append(dict(id=name, title=title, class_name=type(model).__name__,
            initialization='untrained', bridge_gain=gain, basis_std=basis_std,
            note=('Ненулевые KAN-коэффициенты заданы случайно; g=0,25; plastic_gain=0,1. '
                  'Это проверка прохождения сигнала, не результат обучения.' if basis_std else
                  'Общие веса одинаковы у всех режимов и случайно инициализированы. '
                  'g=1 — явно заданная тестовая перемычка.' if gain==1 else
                  'Точная инициализация выбранного класса; новые ветки с нулевым коэффициентом не дают тока.'),
            parameter_count=sum(p.numel() for p in model.parameters()),
            parameters=export_parameters(model), trace={k: packed(v) for k,v in trace.items()}, audit=audit,
            statistics=visualization_statistics(model, trace)))
    for source in sources:
        if hashlib.sha256((source_root/source['path']).read_bytes()).hexdigest()!=source['sha256']:
            raise RuntimeError('Model source changed while the forward was being recorded.')
    return dict(version=1, cfg=asdict(cfg), seed=seed, ids=ids[0].tolist(), text=text,
        provenance='CPU / FP32 / actual production forward / no optimizer / no checkpoint',
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        step_scale=base.step_scale, edge_keys=list(base.edges), variants=variants, sources=sources,
        visualization_scales=dict(
            state=max(.2, .42*max(v['statistics']['max_abs_state'] for v in variants)),
            current=max(1e-6, max(v['statistics']['max_abs_direct_current'] for v in variants)),
            reference='Fixed across all variants, hops and positions. State colours saturate above this scale.'),
        shared=dict(embedding=packed(base.embedding.weight), readout=packed(base.readout),
                    mlp_up=packed(torch.stack([m.up.weight for m in base.neurons])),
                    mlp_down=packed(torch.stack([m.down.weight for m in base.neurons]))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('docs/transport_3d.html'))
    p.add_argument('--neurons',type=int,default=128)
    p.add_argument('--text',default='Nim=417; Nim=')
    p.add_argument('--seed',type=int,default=230923)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--route-seed',type=int,default=0)
    p.add_argument('--text-file',type=Path)
    p.add_argument('--label')
    a=p.parse_args()
    if a.checkpoint:
        if not a.source_root:p.error('--checkpoint requires the frozen --source-root')
        command=[sys.executable,str(Path(__file__).with_name('build_checkpoint_view.py')),
                 '--checkpoint',str(a.checkpoint),'--source-root',str(a.source_root),
                 '--out',str(a.out),'--text',a.text,'--route-seed',str(a.route_seed)]
        if a.text_file:command.extend(['--text-file',str(a.text_file)])
        if a.label:command.extend(['--label',a.label])
        # A fresh interpreter must import the checkpoint's own source tree.
        subprocess.run(command,check=True)
        return
    if a.source_root or a.text_file or a.label:p.error('checkpoint options require --checkpoint')
    payload=build_payload(a.neurons,a.text,a.seed)
    template=Path(__file__).with_name('transport_view.html').read_text()
    if template.count('__TRANSPORT_DATA__')!=1: raise ValueError('invalid viewer template')
    payload['template_sha256']=hashlib.sha256(template.encode()).hexdigest()
    data=json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(template.replace('__TRANSPORT_DATA__',data))
    audit=dict(config=payload['cfg'], seed=a.seed, source_hashes={r['path']:r['sha256'] for r in payload['sources']},
               variants={v['id']:v['audit'] for v in payload['variants']}, html_bytes=a.out.stat().st_size,
               template_sha256=payload['template_sha256'],
               honesty='Fresh 128-wide actual models; no trained checkpoint, no hidden aggregation, no learning claims.')
    a.out.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
