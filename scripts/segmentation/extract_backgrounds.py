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


def process_one_mask(img_path, msk_path, dilation=20):
    img = load_image(img_path)
    msk = load_mask(msk_path)
    kernel = np.ones((dilation, dilation), np.uint8)
    msk = cv2.dilate(msk.astype(np.uint8), kernel)[..., None] > 0  # bool -> uint8 -> bool # BUG: last dimension gone
    img = torch.from_numpy(img)
    msk = torch.from_numpy(msk)
    inv_msk = ~msk
    msked = img * inv_msk  # remove foreground
    return msked, inv_msk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/mobile_stage/dance3')
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--masks_dir', type=str, default='mask')
    parser.add_argument('--bkgd_dir', type=str, default='bkgd')
    parser.add_argument('--image_ext', type=str, default='.jpg')
    parser.add_argument('--mask_ext', type=str, default='.jpg')
    parser.add_argument('--ext', type=str, default='.png')
    parser.add_argument('--jump', type=int, default=10)
    parser.add_argument('--dilation', type=int, default=20)
    args = parser.parse_args()

    cameras = sorted(listdir(join(args.data_root, args.images_dir)))
    os.makedirs(join(args.data_root, args.bkgd_dir), exist_ok=True)
    img_list = {cam: glob.glob(join(args.data_root, args.images_dir, cam, f'*{args.image_ext}')) for cam in cameras}
    msk_list = {cam: [i.replace(args.images_dir, args.masks_dir).replace(args.image_ext, args.mask_ext) for i in v] for cam, v in img_list.items()}  # assuming all cameras has the same number of images

    # load all images
    run(f'mkdir -p {join(args.data_root, args.bkgd_dir)}')
    for cam in cameras:
        ret = parallel_execution(img_list[cam][::args.jump], msk_list[cam][::args.jump], args.dilation, action=process_one_mask, print_progress=True)
        msked = torch.stack([m[0].cpu() for m in ret])  # N, H, W
        inv_msk = torch.stack([m[1].cpu() for m in ret])  # N, H, W
        msked = msked.sum(dim=0)
        inv_msk = inv_msk.sum(dim=0).clip(1e-6)
        bkgd = msked / inv_msk
        bkgd = torch.cat([bkgd, inv_msk >= 1], dim=-1)
        bkgd_path = join(args.data_root, args.bkgd_dir, f'{cam}{args.ext}')
        save_image(bkgd_path, bkgd.numpy())
        log(f'Saving merged background to: {blue(bkgd_path)}')


if __name__ == '__main__':
    main()
