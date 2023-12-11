import os
import cv2
import torch
import argparse
import numpy as np

from tqdm import tqdm
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_image, load_mask, save_image, save_mask, list_to_tensor, tensor_to_list
from easyvolcap.utils.parallel_utils import parallel_execution


def get_bkgd_path(bkgd_dir: str, img_path: str, cam: str):
    # load background image for this camera
    bkgd_path = join(bkgd_dir, f'{cam}.jpg')
    if not os.path.exists(bkgd_path):
        bkgd_path = bkgd_path.replace('.jpg', '.png')
    if not os.path.exists(bkgd_path):
        bkgd_path = join(bkgd_dir, f'{cam}', f'000000.jpg')
    if not os.path.exists(bkgd_path):
        bkgd_path = bkgd_path.replace('.jpg', '.png')
    if not os.path.exists(bkgd_path):
        img = os.path.basename(img_path)
        cam = img.split('.')[1]
        bkgd_path = join(bkgd_dir, f'{cam}', f'000000.jpg')
    if not os.path.exists(bkgd_path):
        bkgd_path = bkgd_path.replace('.jpg', '.png')

    return bkgd_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt_path', type=str, default='/nas/home/xuzhen/weights/backgroundmattev2/torchscript_resnet101_fp32.pth')
    parser.add_argument('--data_root', type=str, default='/nas/home/xuzhen/datasets/xuzhen36/talk')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--mask_dir', type=str, default='bgmtv2')
    parser.add_argument('--bkgd_dir', type=str, default='bkgd')
    parser.add_argument('--mask_ext', type=str, default='.jpg')
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    log(f'Loading backgroundmattev2 model: {args.chkpt_path}')
    model = torch.jit.load(args.chkpt_path, map_location='cuda').eval()
    model.backbone_scale = 0.25
    model.refine_sample_pixels = 80_000

    for cam in sorted(os.listdir(join(args.data_root, args.image_dir))):
        log(f'Processing camera: {cam}')
        # prepare for corresponding mask path
        cam_dir = join(join(args.data_root, args.image_dir), cam)
        cam_mask_dir = join(join(args.data_root, args.mask_dir), cam)
        os.system(f'mkdir -p {cam_mask_dir}')

        # iterate through all images of this camera
        img_list = [join(cam_dir, img) for img in sorted(os.listdir(cam_dir))]  # full path of all input images
        msk_list = [join(cam_mask_dir, os.path.basename(img).replace('.jpg', args.mask_ext)) for img in img_list]  # full path of all output masks
        bkgd = get_bkgd_path(join(args.data_root, args.bkgd_dir), img_list[0], cam)

        # iterate through all unique background images
        log(f'Processing background: {bkgd}, image count: {len(img_list)}')

        # prepare background image tensor
        bgr = list_to_tensor([load_image(bkgd)[..., :3]])

        # iterate through all images of this camera
        n_images = len(img_list)
        for i in tqdm(range(0, n_images, args.chunk_size)):  # 0, 1000, 64
            img_chunk = img_list[i:i + args.chunk_size]
            msk_chunk = msk_list[i:i + args.chunk_size]
            # load all 64 images in parallel

            src = list_to_tensor(parallel_execution(img_chunk, action=load_image))

            # pass through the network in batch
            res = []
            n_images_chunks = len(img_chunk)
            for j in range(0, n_images_chunks, args.batch_size):
                src_chunk = src[j:j + args.batch_size]
                with torch.no_grad():
                    pha, fgr = model(src_chunk, bgr.expand(*src_chunk.shape))[:2]
                    res.append(pha)
            res = torch.cat(res, dim=0)

            # save all 64 image in parallel
            parallel_execution(msk_chunk, tensor_to_list(res), action=save_mask, quality=85)

    log(yellow(f'Matting result saved to {blue(join(args.data_root, args.mask_dir))}'))


if __name__ == '__main__':
    main()
