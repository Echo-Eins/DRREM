"""DRREM event-STDP entry point: train or sample the spiking machine."""
import argparse
import importlib
import sys


def main():
    commands={'train':'scripts.train_spiking_stdp','sample':'scripts.sample_spiking_stdp'}
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=commands,help='use COMMAND --help for its options')
    args=sys.argv[1:]
    if not args:
        parser.print_help()
        return
    command=parser.parse_args(args[:1]).command
    sys.argv=[f'{sys.argv[0]} {command}',*args[1:]]
    importlib.import_module(commands[command]).main()


if __name__=='__main__':
    main()
