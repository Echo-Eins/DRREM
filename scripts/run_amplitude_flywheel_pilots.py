"""Prespecified repair test; reuse the completed identical unit-vector control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def last_dev(path):
    rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    if rows[-1]['event']!='finished':raise RuntimeError('unfinished control')
    return [r for r in rows if 'dev' in r][-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--control-root',type=Path,default=Path('runs/directed_flywheel_20260921/routed_pilots'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    control=last_dev(a.control_root/'decoder_state');off=last_dev(a.control_root/'off')
    status=dict(stage='running',started=time.time(),test_opened=False,
        hypothesis='unit normalization amplifies weak errors; a linear conditioner cannot reconstruct amplitude from an extra scalar',
        control=str(a.control_root/'decoder_state'),no_hint_control=str(a.control_root/'off'),
        calibration=json.loads(a.calibration.read_text()))
    def save():
        path=a.out/'status.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(a.out/'status.json')
    save()
    command=[sys.executable,'-m','scripts.train_amplitude_flywheel','--steps','128','--compile-parts',
        '--out',str(a.out/'amplitude'),'--adapter-lr','3e-4','--calibration',str(a.calibration)]
    status['command']=command;save()
    with (a.out/'amplitude.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status.update(returncode=r.returncode,finished=time.time());save()
    if r.returncode:status['stage']='failed';save();raise RuntimeError('amplitude test failed')
    final=last_dev(a.out/'amplitude')
    # Effective batch, data cursor, ordinary Adam and endpoint must match.
    for c in [control,off]:
        if (c['stage_step'],c['global_step'],c['response_exposures'])!=(final['stage_step'],final['global_step'],final['response_exposures']):
            raise RuntimeError('unmatched controls')
    status.update(stage='finished',final_bpb=final['dev']['final_bpb'],first_bpb=final['dev']['first_bpb'],
        versus_unit=paired_difference(final['dev']['documents'],control['dev']['documents']),
        versus_no_hint=paired_difference(final['dev']['documents'],off['dev']['documents']),
        second_minus_first=paired_difference(final['dev']['documents'],final['dev']['documents'],'final_nats','first_nats'))
    save();print(json.dumps(status),flush=True)


if __name__=='__main__':main()
