import os
import cv2
import argparse
from tqdm import tqdm
from glob import glob
from os.path import join


# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import log, run
from easyvolcap.utils.data_utils import load_unchanged, save_unchanged
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on


def resize_hdri(img_path, out_path, height=16, width=32):
    log(f'loading {img_path}')
    hdri = load_unchanged(img_path)
    log(f'resizing {img_path}')
    hdri = cv2.resize(hdri, (width, height), interpolation=cv2.INTER_AREA)
    log(f'saving {img_path} to {out_path}')
    save_unchanged(out_path, hdri)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/lighting')
    parser.add_argument('--dir_8k', default='8k')
    parser.add_argument('--dir_16x32', default='16x32')
    args = parser.parse_args()
    img_paths = glob(join(args.data_root, args.dir_8k, "*.hdr")) + glob(join(args.data_root, args.dir_8k, "*.exr"))
    out_paths = [join(args.data_root, args.dir_16x32, os.path.basename(img_path)) for img_path in img_paths]
    parallel_execution(img_paths, out_paths, action=resize_hdri, print_progress=True)


if __name__ == '__main__':
    main()
