"""
This file will read all images and generate a video from it
"""

import os
import cv2
import torch
import argparse
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.data_utils import load_image, load_mask, save_image, numpy_to_video, video_to_numpy, numpy_to_list
from easyvolcap.utils.parallel_utils import parallel_execution


def save_one_image(img: np.ndarray, out: str, FH: int = -1, FW: int = -1):
    H, W, C = img.shape
    if FH != -1 and FW != -1 and (H != FH or W != FW):
        img = cv2.resize(img, (FW, FH), interpolation=cv2.INTER_AREA)  # effectively nearest
    save_image(out, img)


@catch_throw
def main():
    args = dotdict()
    args.data_root = 'data/renbody/0013_01'
    args.videos_dir = 'videos_libx265'
    args.images_dir = 'images_libx265'
    args.vcodec = dotdict(default='hevc_cuvid', choices=['hevc_cuvid', 'libx265', 'libx264', 'none'])
    args.hwaccel = dotdict(default='cuda', choices=['cuda', 'none'])
    args.single_channel = False
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    videos_dir = join(args.data_root, args.videos_dir)
    images_dir = join(args.data_root, args.images_dir)
    videos = sorted(os.listdir(videos_dir))

    timer.disabled = False
    for video in videos:
        log(f'Processing video: {blue(video)}')
        timer.record()
        vid = video_to_numpy(join(videos_dir, video), vcodec=args.vcodec, hwaccel=args.hwaccel)
        timer.record('decoding')
        if args.single_channel:
            vid = vid[..., 0][..., None]  # Only save the first channel
        split = video.split('.')[0].split('_')
        N, h, w, C = vid.shape
        x, y, W, H, FW, FH = split[2][1:], split[3][1:], split[4][1:], split[5][1:], split[6][2:], split[7][2:]
        x, y, W, H, FW, FH = int(x), int(y), int(W), int(H), int(FW), int(FH)
        cam = split[0]
        vid = torch.from_numpy(vid)
        full = vid.new_zeros(N, H, W, C)
        full[:, y:y + h, x:x + w] = vid

        images = [join(images_dir, cam, f'{i:06d}.png') for i in range(N)]  # all output images path
        log(f'Number of images: {len(images)}')
        log(f'Output directory: {blue(join(images_dir, cam))}')
        parallel_execution([i for i in full.numpy()], images, FW=FW, FH=FH, action=save_one_image, print_progress=False)


if __name__ == '__main__':
    main()
