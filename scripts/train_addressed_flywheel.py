"""Common full-Adam protocol, with explicit correction-origin addressing."""
import argparse

from drrem.core.addressed_flywheel import AddressedFlywheelMachine
from scripts.train_routed_flywheel import main as train


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--memory-mode',choices=['addressed','off','uniform'],default='addressed')
    p.add_argument('--response-only',action='store_true')
    a,rest=p.parse_known_args()
    train(rest,model_factory=AddressedFlywheelMachine,
          additional_sources=['drrem/core/addressed_flywheel.py','scripts/train_addressed_flywheel.py'],
          factory_kwargs=dict(memory_mode=a.memory_mode,response_only=a.response_only))


if __name__=='__main__':main()
