import os
import argparse
from glob import glob
from os.path import join

# fmt: off
import sys
sys.path.append('.')

from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.console_utils import run, log, run_if_not_exists
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_root', default='/nas/home/xuzhen/datasets/my_zjumocap')
    parser.add_argument('--target_root', default='data/my_zjumocap')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--human', default='my_377')
    args = parser.parse_args()
    args.source_root = join(args.source_root, args.human)
    args.target_root = join(args.target_root, args.human)

    # grab all image files
    log(f'grabbing all image files, only second level or more')
    images = glob(join(args.source_root, 'images', '**', '*.jpg')) + glob(join(args.source_root, 'images', '**', '*.png'))
    mask = glob(join(args.source_root, 'mask', '**', '*.jpg')) + glob(join(args.source_root, 'mask', '**', '*.png'))
    schp = glob(join(args.source_root, 'schp', '**', '*.jpg')) + glob(join(args.source_root, 'schp', '**', '*.png'))

    # grab all smpl params
    log(f'grabbing all smpl parameters files, only the first level')
    smpl_params = glob(join(args.source_root, 'smpl_params', '*'))
    smpl_vertices = glob(join(args.source_root, 'smpl_vertices', '*'))
    smpl_lbs = glob(join(args.source_root, 'smpl_lbs', '*.npy'))  # exclude bweights

    # grad other files
    others = [
        join(args.source_root, 'annots.npy'),
        join(args.source_root, 'intri.yml'),
        join(args.source_root, 'extri.yml'),
    ]

    # merge all files into one list
    all_sources = images + mask + schp + smpl_params + smpl_vertices + smpl_lbs + others
    all_targets = [f.replace(args.source_root, args.target_root) for f in all_sources]

    # run things in parallel for all files
    def copy_one_file(source, target, dry_run=False):
        if not dry_run:
            os.makedirs(os.path.dirname(target), exist_ok=True)
        run_if_not_exists(f'cp {source} {target}', target, dry_run=dry_run)

    parallel_execution(all_sources, all_targets, args.dry_run, action=copy_one_file, num_workers=64, print_progress=True)


if __name__ == '__main__':
    main()
