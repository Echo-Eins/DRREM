"""Same six hops and final-only CE+MTP; a provisional solve feeds a live hint."""
import sys

from drrem.core.split_flywheel import SplitFlywheelMachine
from scripts.train_routed_flywheel import main as train


def main():
    if '--first-weight' in sys.argv:raise ValueError('split experiment fixes final-only supervision')
    train(sys.argv[1:]+['--first-weight','0'],model_factory=SplitFlywheelMachine,
          additional_sources=['drrem/core/split_flywheel.py','scripts/train_split_flywheel.py'])


if __name__=='__main__':main()
