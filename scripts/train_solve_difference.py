from drrem.core.solve_difference import SolveDifferenceMachine
from scripts.train_routed_flywheel import main as train


if __name__=='__main__':
    import sys
    if '--first-weight' in sys.argv:raise ValueError('only the final prediction receives CE+MTP')
    train(sys.argv[1:]+['--first-weight','0'],model_factory=SolveDifferenceMachine,
          additional_sources=['drrem/core/solve_difference.py','scripts/train_solve_difference.py'])
