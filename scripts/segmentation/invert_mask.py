import os
import cv2
import torch
import argparse
from glob import glob
from os.path import join, exists

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_mask


def inv_one_mask(mask_path: str, masks_dir: str, inv_masks_dir: str):
    mask = torch.as_tensor(load_mask(mask_path), dtype=torch.bool)
    inv_mask = 1 - mask  # inverse
    inv_mask_path = mask_path.replace(masks_dir, inv_masks_dir)
    os.makedirs(os.path.dirname(inv_mask_path), exist_ok=True)
    cv2.imwrite(inv_mask_path, (inv_mask.numpy() * 255).astype(np.uint8))
    # print(f'Inverted mask {mask_path} -> {inv_mask_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/zju/ip412')
    parser.add_argument('--masks_dir', type=str, default='merged_mask')
    parser.add_argument('--inv_masks_dir', type=str, default='inv_maskes')
    args = parser.parse_args()

    args.masks_dir = join(args.data_root, args.masks_dir)
    args.inv_masks_dir = join(args.data_root, args.inv_masks_dir)
    maskes_paths = glob(join(args.masks_dir, '**', '*.png'), recursive=True)
    parallel_execution(maskes_paths, args.masks_dir, args.inv_masks_dir, action=inv_one_mask, print_progress=True)


if __name__ == '__main__':
    main()
