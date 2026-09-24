from drrem.core.arrival_radial import ArrivalRadialMachine
from scripts.train_routed_flywheel import main as train


if __name__=='__main__':
    import sys
    if '--first-weight' in sys.argv:raise ValueError('one-solve control')
    train(sys.argv[1:]+['--first-weight','0'],model_factory=ArrivalRadialMachine,
          additional_sources=['drrem/core/arrival_radial.py','drrem/core/radial_transport.py','scripts/train_arrival_radial.py'])
