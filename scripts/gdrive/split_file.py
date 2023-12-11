import os
import argparse
from easyvolcap.utils.console_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/nas/home/xuzhen/datasets/vhulls.tar.gz')
parser.add_argument('--chunk', default=1024)  # 2GB small files
args = parser.parse_args()

input = args.input
output = args.input + '.split'
prefix = os.path.basename(args.input) + '.'

run(f'mkdir -p {output}')
run(f'split -d -b {args.chunk}M {input} {output}/{prefix}') # prepare for uploading
# for f in os.listdir(output):
    # run(f'mv {output}/{f} {output}/{prefix}{f}')
