import os
import argparse
from os.path import join
from easyvolcap.utils.console_utils import *


def walk(src, tar, fuzzy: bool = False):
    print(f"walking: {os.getcwd()}")
    for d in os.listdir():
        if os.path.isdir(d):
            prev = os.getcwd()
            os.chdir(d)
            walk(src, tar, fuzzy)
            os.chdir(prev)  # prevent symlink from messing up directory structure
        condition = (src in d) if fuzzy else (d == src)
        if condition:
            run(f'mv {d} {d.replace(src, tar)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('tar', type=str)
    parser.add_argument('--only', type=str, nargs='*',
                        default=['record', 'trained_model', 'result', 'novel_view', 'novel_pose', 'novel_light', 'pose_sequence', 'animation', 'deformation'],
                        )
    parser.add_argument('--fuzzy', action='store_true')
    args = parser.parse_args()
    prev = os.getcwd()
    for d in args.only:
        d = join('data', d)
        if not os.path.isdir(d): continue
        os.chdir(d)
        walk(args.src, args.tar, args.fuzzy)
        os.chdir(prev)


if __name__ == "__main__":
    main()
