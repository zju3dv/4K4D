# Robust video matting cannot produce clear or robust mask, but it's good at separating the foreground from the background
# Remove the moving people from the foreground, maybe we should consider taking the median? hard to implement a batched median algorithm with uneven pixel count... so stucking with mean for now

import os
import cv2
import glob
import torch
import argparse
import numpy as np
from os import listdir
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_mask, save_mask, load_image, save_image
from easyvolcap.utils.parallel_utils import parallel_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/volcano/skateboard')
    parser.add_argument('--source_dir', type=str, default='bkgd_d75_j1')  # main bkgd
    parser.add_argument('--paste_dir', type=str, default='bkgd_d20_j1')  # fill black regions with content here
    parser.add_argument('--bkgd_dir', type=str, default='bkgd')
    args = parser.parse_args()

    for img in sorted(os.listdir(join(args.data_root, args.source_dir))):
        img_path = join(args.data_root, args.source_dir, img)
        pst_path = join(args.data_root, args.paste_dir, img)
        out_path = join(args.data_root, args.bkgd_dir, img)
        img = load_image(img_path)
        img = torch.from_numpy(img)

        pst = load_image(pst_path)
        pst = torch.from_numpy(pst)

        img[..., :3] = img[..., :3] * img[..., 3:] + pst[..., :3] * (1 - img[..., 3:])
        img[..., 3:] = img[..., 3:] * img[..., 3:] + pst[..., 3:] * (1 - img[..., 3:])

        save_image(out_path, img.numpy())
        log(yellow(f'Merged bkgd saved to {blue(out_path)}'))


if __name__ == '__main__':
    main()
