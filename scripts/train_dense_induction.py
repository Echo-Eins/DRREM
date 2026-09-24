from drrem.core.dense_induction import DenseInductionMachine
from scripts.train_routed_flywheel import main as train


if __name__=='__main__':
    import sys
    if '--first-weight' in sys.argv:raise ValueError('only final mixture receives next-byte CE')
    train(sys.argv[1:]+['--first-weight','0'],model_factory=DenseInductionMachine,
          additional_sources=['drrem/core/induction.py','drrem/core/radial_transport.py',
              'drrem/core/induction_transport.py','drrem/core/dense_induction.py','scripts/train_dense_induction.py'])
