"""Bounded FineWeb experiment: useful energy work versus neuron-current fatigue."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.energy_budget import budgeted_descent
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, evaluate
from scripts.summarize_fineweb import paired


class BudgetEnergyMachine(EquilibriumEnergyTransportMachine):
    def __init__(self, cfg, fraction, recharge):
        super().__init__(cfg)
        self.fraction, self.recharge = fraction, recharge
        self.base_cache = None
        self.audit = []

    def solve_energy(self, states, return_trace=False):
        if return_trace:
            raise ValueError('use the recorded budget audit')
        with torch.autocast(states[0].device.type, enabled=False):
            anchors, scales, precision, operators = self.energy_context(states)
            key = tuple(p._version for p in [self.precision_bias]+[e.weight for e in self.edges.values()])
            if self.base_cache is None or self.base_cache[0] != key:
                base, p0 = self.shared_hessian(operators)
                self.base_cache = key, base, p0
            _, base, p0 = self.base_cache
            initial = torch.cat(anchors, -1)
            rhs = torch.cat([p*a for p,a in zip(precision, anchors)], -1)
            delta = torch.cat([p-p0[i] for i,p in enumerate(precision)], -1)
            width = rhs.shape[-1]
            solution, audit = budgeted_descent(base, delta.reshape(-1,width), rhs.reshape(-1,width),
                initial.reshape(-1,width), layers=self.cfg.layers, steps=16, lifetime=4,
                fraction=self.fraction, recharge=self.recharge)
            energy = torch.stack([r['energy'] for r in audit['trace']])
            self.audit.append(dict(rows=rhs.numel()//width, iterations=len(audit['trace'])-1,
                energy_reduction=float((energy[0]-energy[-1]).mean()),
                max_energy_increase=float((energy[1:]-energy[:-1]).max()) if len(energy)>1 else 0.,
                mean_neuron_updates=float(audit['updates'].float().mean()),
                mean_relative_residual=float(audit['relative_residual'].mean()),
                max_relative_residual=float(audit['relative_residual'].max())))
            return tuple(x*s for x,s in zip(solution.view_as(rhs).split(self.cfg.neurons,-1), scales))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--windows-per-document', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(args.parent, map_location='cpu', weights_only=False, mmap=True)
    corpus = FineWebBytes(DEFAULT_CACHE)
    plan = dict(ck['protocol']['dev'])
    counts = Counter()
    units = []
    for unit in plan['units']:
        doc = unit[0]
        if counts[doc] < args.windows_per_document:
            units.append(unit)
            counts[doc] += 1
    plan['units'] = units
    result = dict(parent_sha256=digest(args.parent), data_plan=plan,
        source_hashes={path:digest(path) for path in ['drrem/core/energy_budget.py','scripts/probe_energy_budget.py']},
        scope='No parameter updates; first windows of the same dev documents; preserved content, fatigue only on corrective currents. Dense products still paid.',
        arms={})
    for name, fraction, recharge in [('equilibrium',1.,0.), ('fatigue_no_recharge',1.,0.),
                                     ('recharge_dense',1.,4.), ('recharge_quarter',.25,4.)]:
        cfg = CausalTransportConfig(**ck['protocol']['model'])
        model = EquilibriumEnergyTransportMachine(cfg) if name == 'equilibrium' else BudgetEnergyMachine(cfg, fraction, recharge)
        model.cuda().eval().load_state_dict(ck['model'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.monotonic()
        value = evaluate(model, corpus, plan)
        torch.cuda.synchronize()
        row = dict(dev=value, seconds=time.monotonic()-begin,
            peak_gib=torch.cuda.max_memory_allocated()/2**30,
            refinement_audit=getattr(model, 'audit', None))
        if name != 'equilibrium':
            row['vs_equilibrium'] = paired(value['documents'], result['arms']['equilibrium']['dev']['documents'])
        result['arms'][name] = row
        args.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(arm=name, bpb=value['bpb'], seconds=row['seconds'], comparison=row.get('vs_equilibrium'))), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
