"""
This file will read all images and generate a video from it
"""

import os
import cv2
import torch
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_image, load_mask, save_image, save_mask, numpy_to_video, video_to_numpy, numpy_to_list
from easyvolcap.utils.parallel_utils import parallel_execution


def save_one_image(img: np.ndarray, out: str, FH: int = -1, FW: int = -1, thresh: float = 12, denoise: bool = False):
    if denoise: msk = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8), None, 10, 10, 7, 21)
    else: msk = img
    msk = msk.sum(axis=-1, keepdims=True) > thresh
    H, W, C = msk.shape
    if FH != -1 and FW != -1 and (H != FH or W != FW):
        msk = cv2.resize(msk.astype(np.uint8), (FW, FH), interpolation=cv2.INTER_AREA)
    save_mask(out, msk)


@catch_throw
def main():
    args = dotdict()
    args.data_root = 'data/renbody/0013_01'
    args.videos_dir = 'videos_libx265'
    args.masks_dir = 'masks_libx265'
    args.thresh = 12
    args.denoise = False
    args.vcodec = dotdict(default='hevc_cuvid', choices=['hevc_cuvid', 'libx265', 'libx264', 'none'])
    args.hwaccel = dotdict(default='cuda', choices=['cuda', 'none'])
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    videos_dir = join(args.data_root, args.videos_dir)
    masks_dir = join(args.data_root, args.masks_dir)
    videos = sorted(os.listdir(videos_dir))

    for video in videos:
        log(f'Processing video: {blue(video)}')
        vid = video_to_numpy(join(videos_dir, video), vcodec=args.vcodec, hwaccel=args.hwaccel)
        split = video.split('.')[0].split('_')
        N, h, w, C = vid.shape
        x, y, W, H, FW, FH = split[2][1:], split[3][1:], split[4][1:], split[5][1:], split[6][2:], split[7][2:]
        x, y, W, H, FW, FH = int(x), int(y), int(W), int(H), int(FW), int(FH)
        cam = split[0]
        vid = torch.from_numpy(vid)
        full = vid.new_zeros(N, H, W, C)
        full[:, y:y + h, x:x + w] = vid

        masks = [join(masks_dir, cam, f'{i:06d}.png') for i in range(N)]  # all output images path
        log(f'Number of masks: {len(masks)}')
        log(f'Output directory: {blue(join(masks_dir, cam))}')
        parallel_execution([i for i in full.numpy()], masks, thresh=args.thresh, FW=FW, FH=FH, denoise=args.denoise, action=save_one_image, print_progress=True)


if __name__ == '__main__':
    main()
