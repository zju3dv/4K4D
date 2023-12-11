# This script will read all experiments in the data/trained_model folder
# And remove numbered files if the number of files is larger than 3
# Possibly ignoring some of the files
from easyvolcap.utils.console_utils import *

import argparse
from glob import glob


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/trained_model')
    parser.add_argument('--ext', default='.pt')
    parser.add_argument('--count', default=3, type=int)
    parser.add_argument('--remove_only_if_contains', default='latest.pt')
    parser.add_argument('--skip', default=['training_r4'], nargs='*')
    args = parser.parse_args()

    to_remove_all = []
    for exp in os.listdir(args.input):
        if exp in args.skip:
            continue
        files = os.listdir(join(args.input, exp))
        files = [f for f in files if f.endswith(args.ext)]
        files = sorted(files, key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 99999)
        if args.remove_only_if_contains not in files:
            continue
        if len(files) <= args.count:
            continue
        to_remove = files[:-args.count]
        to_remove_all += [join(args.input, exp, f) for f in to_remove]

    log(to_remove_all)
    choice = input('remove these files? (yes/no)')
    if choice.lower() == 'yes' or choice.lower() == 'y':
        run(f'rm {" ".join(to_remove_all)}')


if __name__ == '__main__':
    main()
