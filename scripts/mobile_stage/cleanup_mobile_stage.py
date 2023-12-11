import os
import argparse
from os.path import join
from easyvolcap.utils.console_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/mobile_stage')
parser.add_argument('--output', default='/nas/home/xuzhen/datasets/mobile_stage_clean')
parser.add_argument('--humans', default=['xuzhen', 'purple', 'black', 'white', 'dance3', 'female'], nargs='+')
parser.add_argument('--contents', default=['images', 'mask', 'rvm', 'schp', 'bgmtv2', 'extri.yml', 'intri.yml'], nargs='+')
args = parser.parse_args()

for human in args.humans:
    run(f'mkdir -p {join(args.output, human)}')
    for content in args.contents:
        run(f'cp -r {join(args.input, human, content)} {join(args.output, human, content)}', skip_failed=True)
