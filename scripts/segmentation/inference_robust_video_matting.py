import os
import torch
import argparse
import numpy as np
from os import listdir
from os.path import join

# fmt: off
import sys

sys.path.append('.')
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_image, load_mask, save_image, save_mask, list_to_tensor, tensor_to_list
from easyvolcap.utils.parallel_utils import parallel_execution

# NOTE: this hacky is only effective if you're running the script from the root directory of this repository
# By default, you can't import other files in the parent directory. When importing a file, Python only searches the directory that the entry-point script is running from and sys.path which includes locations such as the package installation directory (it's actually a little more complex than this, but this covers most cases).
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/nas/home/xuzhen/datasets/xuzhen36/talk')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--mask_dir', type=str, default='rvm')
    parser.add_argument('--mask_ext', type=str, default='.jpg')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    log(f'Loading RobustVideoMatting model')
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda().eval()  # or "resnet50"

    cameras = sorted(listdir(join(args.data_root, args.image_dir)))
    frame_list = sorted(listdir(join(join(args.data_root, args.image_dir), cameras[0])))  # assuming all cameras has the same number of images
    [os.system(f'mkdir -p {join(join(args.data_root, args.mask_dir), camera)}') for camera in cameras]  # create output directory

    n_rec = 4  # number of recurrent states
    rec = [None] * n_rec  # recurrent states
    for frame in tqdm(frame_list):
        img_list = [join(join(args.data_root, args.image_dir), camera, frame) for camera in cameras]
        msk_list = [join(join(args.data_root, args.mask_dir), camera, frame.replace('.jpg', args.mask_ext)) for camera in cameras]
        src = list_to_tensor(parallel_execution(img_list, action=load_image))  # copy to GPU non-blocking

        # pass through the network in batch
        msk_results = []  # background segmentation
        rec_results = []  # recurrent states
        n_cameras = len(img_list)
        for j in range(0, n_cameras, args.batch_size):
            src_chunk = src[j:j + args.batch_size]
            rec_chunk = [r[j:j + args.batch_size] for r in rec] if rec[0] is not None else rec
            # pass through the network in batch (all cameras of one frame)
            with torch.no_grad():
                fgr, msk, *rec_chunk = model(src_chunk, *rec_chunk, downsample_ratio=0.25)
                msk_results.append(msk)
                rec_results.append(rec_chunk)
        rec = [torch.cat([r[i] for r in rec_results], dim=0) for i in range(n_rec)]  # merge updated recurrent states respectively
        msk = torch.cat(msk_results, dim=0)  # merge output mask

        # save images at this frame in parallel
        parallel_execution(msk_list, tensor_to_list(msk), action=save_mask, quality=85)

    log(yellow(f'Matting result saved to {blue(join(args.data_root, args.mask_dir))}'))


if __name__ == '__main__':
    main()
