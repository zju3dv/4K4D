import os
import argparse
from easyvolcap.utils.console_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='/nas/home/xuzhen/datasets/enerf_outdoor_uploading')
parser.add_argument('--prefix', default='enerf_outdoor.tar.gz.')
args = parser.parse_args()


for f in os.listdir(args.input):
    run(f'mv {args.input}/{f} {args.input}/{args.prefix}{f}')
