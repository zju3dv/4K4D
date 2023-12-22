# This file will read all images and generate a video from it

import os
import cv2
import torch
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_image, load_mask, numpy_to_video, save_image
from easyvolcap.utils.parallel_utils import parallel_execution


def mask_one_image(input: str, mask: str):
    img = load_image(input)
    msk = load_mask(mask)
    img[msk[..., 0] < 0.5] = 0  # fill background with blackness
    img = (img.clip(0, 1) * 255).astype(np.uint8)

    bbox = cv2.boundingRect((msk > 0.5).astype(np.uint8))
    return img, bbox  # H, W, 3 and 4


@catch_throw
def main():
    parser = argparse.ArgumentParser(description='Compress images after masking the background region')
    parser.add_argument('--data_root', default='data/renbody/0013_01')
    parser.add_argument('--images_dir', default='images_calib')
    parser.add_argument('--masks_dir', default='masks')
    parser.add_argument('--video_dir', default='images_compressed')
    args = parser.parse_args()

    img_dir = join(args.data_root, args.images_dir)
    cams = sorted(os.listdir(img_dir))
    for cam in cams:
        cam_dir = join(img_dir, cam)
        images = sorted(os.listdir(cam_dir))
        images = [join(cam_dir, i) for i in images]
        masks = [i.replace(args.images_dir, args.masks_dir) for i in images]

        imgs, bbox = zip(*parallel_execution(images, masks, action=mask_one_image, print_progress=True))
        imgs = torch.as_tensor(np.stack(imgs))  # N, H, W, 3, as uint8
        bbox = torch.as_tensor(np.stack(bbox))  # N, 4
        bbox[:, 2:] += bbox[:, :2]  # x, y, x, y
        N, H, W, C = imgs.shape
        x, y = bbox[:, 0].min(), bbox[:, 1].min()
        w, h = bbox[:, 2].max() - x, bbox[:, 3].max() - y
        imgs = imgs[:, y:y + h, x:x + w]  # N, H, W, 3
        output = join(args.data_root, args.video_dir, f'{cam}_N{N}_x{x}_y{y}_W{W}_H{H}.png')
        save_image(imgs.numpy(), output, png_compression=9)  # Convert the numpy array to video


if __name__ == '__main__':
    main()
