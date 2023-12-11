"""
This file will read content from a user specified list of datasets
Link them as if they are continuous
The user can also specify other files to be concatenated
By default, the first sequence's camera parameter is used
"""

import argparse
from easyvolcap.utils.console_utils import *


def link(a: str, b: str):
    return run(f'ln -s {a} {b}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', type=str, default='data/renbody')
    parser.add_argument('--path_list', type=str, nargs='+', default=[
        '0013_01',
        '0013_02',
        '0013_03',
        '0013_04',
        '0013_05',
        '0013_06',
    ], help='The first dataset will be used as skeleton')
    parser.add_argument('--output_path', type=str, default='data/renbody/0013_dance')
    parser.add_argument('--concat_cam_dirs', type=str, nargs='+', default=[
        'images',
        'images_calib',
        'mask',
        'masks',
        'maskes',
        'schp',
        'cameras',
    ], help='These folders will be inspected for camera dirs, where images are stored')
    parser.add_argument('--concat_raw_dirs', type=str, nargs='+', default=[
        'vhulls',
        'surfs',
    ], help='These folders will be inspected for images directly')
    parser.add_argument('--replace_paths', type=str, nargs='+', default=[
        'intri.yml',
        'extri.yml',
        'bkgd',
    ], help='Will use the first dataset\'s value for these paths')
    parser.add_argument('--chunk_size', type=int, default=96)
    args = parser.parse_args()

    run(f'mkdir -p {args.output_path}')
    for i, path in enumerate(args.path_list):

        path = join(args.data_root, path)  # the sequence root
        for f in os.listdir(path):

            # Handle replaced values (camera parameters)
            if i == 0:
                if f in args.replace_paths:
                    src = join(path, f)
                    tar = join(args.output_path, f)
                    src = os.path.relpath(src, os.path.dirname(tar))  # find linking path
                    run(f'ln -s {src} {tar}')

            # Handle raw parameters
            if f in args.concat_raw_dirs:
                run(f'mkdir -p {join(args.output_path, f)}', quite=True)  # create empty dir if necessary
                idx = len(os.listdir(join(args.output_path, f)))  # starting index of the linking
                for j, p in enumerate(tqdm(sorted(os.listdir(join(path, f))), desc=join(path, f))):
                    ext = os.path.splitext(p)[-1]
                    src = join(path, f, p)
                    tar = join(args.output_path, f, f"{j+idx:06d}{ext}")
                    src = os.path.relpath(src, os.path.dirname(tar))  # find linking path
                    run(f'ln -s {src} {tar}', quite=True)

            # Handle camera parameters
            if f in args.concat_cam_dirs:
                for c in os.listdir(join(path, f)):
                    run(f'mkdir -p {join(args.output_path, f, c)}', quite=True)  # create empty dir if necessary
                    idx = len(os.listdir(join(args.output_path, f, c)))  # starting index of the linking
                    for j, p in enumerate(tqdm(sorted(os.listdir(join(path, f, c))), desc=join(path, f))):
                        ext = os.path.splitext(p)[-1]
                        src = join(path, f, c, p)
                        tar = join(args.output_path, f, c, f"{j+idx:06d}{ext}")
                        src = os.path.relpath(src, os.path.dirname(tar))  # find linking path
                        run(f'ln -s {src} {tar}', quite=True)


if __name__ == '__main__':
    main()
