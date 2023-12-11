import os
import argparse
import subprocess
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='/home/sida/xuzhen/misc/vhulls')
parser.add_argument('--name', default='vhulls.tar.gz')  # will use this to filter output
args = parser.parse_args()

pattern = 'gdrive_linux_amd64 download "{id}" --path "{output}"'
exp = '10O6GRKYH47p8vE_izmTy5QQEQCX69NQm'
ids = read(f"gdrive_linux_amd64 list -s 2 | grep {args.name}")
ids = [i[:len(exp)] for i in ids.split('\n') if args.name in i]


def download(id):
    run(pattern.format(id=id, output=args.output))


run(f'mkdir -p {args.output}')
# Will have finished
parallel_execution(ids, action=download)
