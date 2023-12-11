"""
What do we need from the mobile stage datasets?

1. images folder for all camearas (required) -> images/00...35
2. camera calibration (required) -> intri.yml extri.yml
3. segmentation (optional) -> schp mask rvm bgmt
4. lbs metadata (optional) -> lbs
5. pose estimation (optional) -> params vertices easymocap
6. packaged camera calibration (optional) -> annots.npy
"""

import os
import argparse
from glob import glob
from os.path import join

# fmt: off
import sys

sys.path.append('.')
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log, run, run_if_not_exists
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on


def glob_everything(root, ext=''):
    log(f'globbing directory/file: {root}')
    globbed = glob(join(root, f'*{ext}')) + glob(join(root, '**', f'*{ext}')) + glob(join(root))
    globbed = [f for f in globbed if os.path.splitext(f)[1]]  # simply determine whether it's a file by the extension
    return globbed


task_tree = dotdict()  # a copy task tree
task_tree.images = ['images']
task_tree.calib = ['intri.yml', 'extri.yml', 'annots.npy']
task_tree.segment = ['schp', 'mask', 'rvm', 'bgmt', 'bkgd']
task_tree.lbs = ['params', 'vertices', 'lbs', 'easymocap']
required = ['images', 'extri.yml', 'intri.yml']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', default='/nas/home/xuzhen/datasets/xuzhen36')
    parser.add_argument('--target_root', default='/nas/home/xuzhen/datasets/mobile_stage')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--human', default='talk')
    args = parser.parse_args()
    args.source_root = join(args.source_root, args.human)
    args.target_root = join(args.target_root, args.human)

    # recursively traverse the file tree with task tree specification
    def traverse(tree):
        if isinstance(tree, list):  # leaf node
            for key in tree:
                yield key, glob_everything(join(args.source_root, key))
        else:
            for key in tree:
                yield from traverse(tree[key])
    globbed = {k: v for k, v in traverse(task_tree)}

    # merge all files into one list
    # log_file = join(args.target_root, 'cleanup.log')
    # os.makedirs(os.path.dirname(log_file), exist_ok=True)
    all_sources = []
    for k, v in globbed.items():
        if not len(v):
            if k not in required:
                log(f'⚠️ empty: {k}', 'yellow')
            else:
                log(f'❌ empty: {k}', 'red')
        else:
            log(f'✅ found: {k}', 'green')
        all_sources += v

    # construct the results list
    all_targets = [f.replace(args.source_root, args.target_root) for f in all_sources]

    # run things in parallel for all files
    def copy_one_file(source, target, dry_run=False):
        if not dry_run:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        run_if_not_exists(f'cp {source} {target}', target, dry_run=dry_run, quite=True)

    parallel_execution(all_sources, all_targets, args.dry_run, action=copy_one_file, num_workers=64, print_progress=True)


if __name__ == '__main__':
    main()
