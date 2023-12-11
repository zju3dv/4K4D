# Robust video matting cannot produce clear or robust mask, but it's good at separating the foreground from the background
# While the SCHP might introduce other people into the final mask
# The best solution would be to tune SCHP to select only one of the people in the scene
# But temporarily we just want a working example, thus we're merging the two masks

# Binarize the RVM result, dilate a bit, then take intersection with the SCHP mask (mask instead of the part parsing)

import os
import cv2
import glob
import argparse
import numpy as np
from os import listdir
from os.path import join

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.data_utils import load_mask, save_mask, load_image, save_image
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on

dilation = 20


def process_one_mask(rvm_path, schp_path, out_path):
    rvm = load_mask(rvm_path)
    schp = load_image(schp_path)
    rvm = cv2.dilate(rvm.astype(np.uint8), np.ones((dilation, dilation), np.uint8))[..., None] > 0
    out = rvm * schp
    save_image(out_path, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/zju/ip412')
    parser.add_argument('--rvm_dir', type=str, default='rvm')
    parser.add_argument('--schp_dir', type=str, default='mask')
    parser.add_argument('--rvm_ext', type=str, default='.jpg')
    parser.add_argument('--schp_ext', type=str, default='.png')
    parser.add_argument('--out_dir', type=str, default='merged_mask')
    args = parser.parse_args()

    cameras = sorted(listdir(join(args.data_root, args.rvm_dir)))
    [os.system(f'mkdir -p {join(join(args.data_root, args.out_dir), camera)}') for camera in cameras]  # create output directory
    rvm_list = glob.glob(join(args.data_root, args.rvm_dir, '**', f'*{args.rvm_ext}'))
    schp_list = [k.replace(args.rvm_dir, args.schp_dir) for k in rvm_list]  # assuming all cameras has the same number of images
    if args.rvm_ext != args.schp_ext: schp_list = [k.replace(args.rvm_ext, args.schp_ext) for k in schp_list]
    out_list = [k.replace(args.rvm_dir, args.out_dir) for k in rvm_list]  # assuming all cameras has the same number of images

    parallel_execution(rvm_list, schp_list, out_list, action=process_one_mask, print_progress=True)


if __name__ == '__main__':
    main()
