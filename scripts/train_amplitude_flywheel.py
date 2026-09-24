"""Equal-capacity unit-vector versus amplitude-preserving error conditioning."""
import argparse
import json
from pathlib import Path

from drrem.core.amplitude_flywheel import AmplitudeFlywheelMachine
from drrem.data.protocol import file_digest
from scripts.train_semantic_flywheel import DEFAULT_PARENT
from scripts.train_routed_flywheel import main as train


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--packet-mode',choices=['unit','amplitude'],default='amplitude')
    a,rest=p.parse_known_args();profile=json.loads(a.calibration.read_text())
    if profile['parent_sha256']!=file_digest(DEFAULT_PARENT):raise ValueError('calibration belongs to a different parent')
    if '--use-route' in rest:raise ValueError('decoder-credit calibration cannot normalize source-space route credit')
    train(rest,model_factory=AmplitudeFlywheelMachine,
        additional_sources=['drrem/core/amplitude_flywheel.py','scripts/train_amplitude_flywheel.py','scripts/calibrate_credit_scale.py'],
        factory_kwargs=dict(packet_mode=a.packet_mode,credit_scales=profile['credit_scales']))


if __name__=='__main__':main()
