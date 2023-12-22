# This file will read all images and generate a video from it

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_image, load_mask, numpy_to_video, load_unchanged
from easyvolcap.utils.parallel_utils import parallel_execution


def mask_one_image(input: str, mask: str, ratio: float = 1.0, thresh: float = 0.5):
    img = load_image(input, ratio)
    msk = load_mask(mask, ratio)
    img[msk[..., 0] < thresh] = 0  # fill background with blackness
    img = (img.clip(0, 1) * 255).astype(np.uint8)
    bbox = cv2.boundingRect((msk > 0.5).astype(np.uint8))
    return img, bbox  # H, W, 3 and 4


def load_one_image(input: str, mask: str = None, ratio: float = 1.0, thresh: float = 0.5):
    img = load_unchanged(input, ratio)
    img = ((img / np.iinfo(img.dtype).max).clip(0, 1) * 255).astype(np.uint8)
    H, W = img.shape[:2]
    return img, (0, 0, W, H)  # H, W, 3 and 4


def load_one_mask(input: str, mask: str = None, ratio: float = 1.0, thresh: float = 0.5):
    msk = load_unchanged(mask, ratio)
    msk = ((msk / np.iinfo(msk.dtype).max).clip(0, 1) * 255).astype(np.uint8)
    H, W = msk.shape[:2]
    if msk.ndim == 2: msk = msk[..., None]
    full = np.zeros((H, W, 3), dtype=np.uint8)
    full[..., :1] = msk[..., :1]  # only first channel of mask
    return full, (0, 0, W, H)  # H, W, 3 and 4


@catch_throw
def main():
    parser = argparse.ArgumentParser(description='Merge images into a video after masking the background region')
    parser.add_argument('--data_root', default='data/renbody/0013_01')
    parser.add_argument('--images_dir', default='images_calib')
    parser.add_argument('--masks_dir', default='masks')
    parser.add_argument('--videos_dir', default='videos_libx265')
    parser.add_argument('--full_image', action='store_true')
    parser.add_argument('--full_mask', action='store_true')
    parser.add_argument('--crf', type=int, default=25)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--thresh', type=float, default=0.1)
    parser.add_argument('--n_frames', type=int, default=-1)
    args = parser.parse_args()

    img_dir = join(args.data_root, args.images_dir)
    cams = sorted(os.listdir(img_dir))

    # Prepare resizing ratios
    ratios = args.ratio  # note the 's'
    if isinstance(ratios, list): ratios = {c: r for r, c in zip(ratios, cams)}
    if not isinstance(ratios, dict): ratios = {c: ratios for c in cams}

    for cam in cams:
        log(f'Processing camera: {cam}')
        ratio = ratios[cam]
        cam_dir = join(img_dir, cam)
        images = sorted(os.listdir(cam_dir))
        if args.n_frames > 0: images = images[:args.n_frames]
        images = [join(cam_dir, i) for i in images]
        masks = [i.replace(args.images_dir, args.masks_dir) for i in images]
        if not exists(masks[0]): masks = [i.replace('.jpg', '.png') for i in masks]
        if not exists(masks[0]): masks = [i.replace('.png', '.jpg') for i in masks]

        im = Image.open(images[0])
        FW, FH = im.width, im.height  # Full width and full height

        if args.full_image: action = load_one_image
        elif args.full_mask: action = load_one_mask
        else: action = mask_one_image

        imgs, bbox = zip(*parallel_execution(images, masks, ratio=ratio, thresh=args.thresh, action=action, print_progress=True))
        if args.full_image or args.full_mask:
            N = len(imgs)
            H, W, C = imgs[0].shape
            x, y, w, h = 0, 0, W, H
        else:
            imgs = torch.as_tensor(np.stack(imgs))  # N, H, W, 3, as uint8
            bbox = torch.as_tensor(np.stack(bbox))  # N, 4
            bbox[:, 2:] += bbox[:, :2]  # x, y, x, y
            N, H, W, C = imgs.shape
            x, y = bbox[:, 0].min(), bbox[:, 1].min()
            w, h = bbox[:, 2].max() - x, bbox[:, 3].max() - y
            imgs = imgs[:, y:y + h, x:x + w]  # N, H, W, 3
            imgs = imgs.numpy()
        output = join(args.data_root, args.videos_dir, f'{cam}_N{N}_x{x}_y{y}_W{W}_H{H}_FW{FW}_FH{FH}.mp4')
        log(f'Output video: {blue(output)}')
        numpy_to_video(imgs, output, crf=args.crf)  # Convert the numpy array to video


if __name__ == '__main__':
    main()
