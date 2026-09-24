"""Dense all-level skips, same parent and ordinary global Adam protocol."""
import argparse

from drrem.core.radial_transport import RadialTransportMachine
from scripts.train_routed_flywheel import main as train


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--radial-mode',choices=['none','dense_field','dense_state','reciprocal'],required=True)
    a,rest=p.parse_known_args()
    if '--first-weight' in rest:raise ValueError('this is a one-solve experiment')
    train(rest+['--first-weight','0'],model_factory=RadialTransportMachine,
          additional_sources=['drrem/core/radial_transport.py','scripts/train_radial_transport.py'],
          factory_kwargs=dict(radial_mode=a.radial_mode))


if __name__=='__main__':main()
