import os
import argparse
from glob import glob
from functools import reduce
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, load_unchanged, save_unchanged


def resize_image(src: str, tar: str, ratio: float = 0.25):
    log(f'{src} -> {tar}')
    img = load_unchanged(src, ratio)
    save_unchanged(tar, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/crab/images')
    parser.add_argument('--output', default='data/crab/images_4')
    parser.add_argument('--ratio', type=float, default=0.25)
    parser.add_argument('--exts', nargs='+', default=['.png', '.jpg', '.PNG', '.JPG'])
    args = parser.parse_args()

    files = [glob(f'{args.input}/**/*{ext}', recursive=True) for ext in args.exts]
    files = reduce(lambda x, y: x + y, files)
    outs = [f.replace(args.input, args.output) for f in files]

    parallel_execution(files, outs, args.ratio, action=resize_image, print_progress=True)


if __name__ == '__main__':
    main()
