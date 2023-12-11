import os
import cv2
import torch
import argparse
import numpy as npf
from functools import partial
from os.path import join, exists, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_unchanged_image, to_cuda, to_tensor, to_cpu, to_numpy, save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/renbody')
    parser.add_argument('--scenes', type=str, nargs='+', default=['0013_06'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--jpeg_quality', type=int, default=80)
    args = parser.parse_args()
    return args


def calib_color(img_ori: torch.Tensor, bgr_sol: torch.Tensor):
    img_ori = img_ori.float()  # removable variables
    bgr_sol = bgr_sol.float()  # removable variables
    rs = []
    for i in range(3):
        channel = img_ori[..., i]
        X = torch.stack([channel**2, channel, torch.ones_like(channel)], dim=-1)  # B, H, W ,3
        y = (bgr_sol[..., 2 - i, :][..., None, None, :] * X).sum(dim=-1)  # B, 3 @ B, H, W, 3
        rs.append(y)
    rs_img = torch.stack(rs, dim=-1)
    return (rs_img.clip(0, 255)).to(torch.uint8)  # this will block...


@catch_throw
def main(args):
    for scene in tqdm(args.scenes):
        extri_path, intri_path = join(args.input, scene, 'extri.yml'), join(args.input, scene, 'intri.yml')
        cams = read_camera(intri_path, extri_path)
        basenames = sorted(os.listdir(join(args.input, scene, 'images')))
        imps = [[] for _ in range(len(basenames))]
        ccms = [[] for _ in range(len(basenames))]
        for idx, cam in enumerate(basenames):
            frames = sorted(os.listdir(join(args.input, scene, 'images', cam)))
            for frame in frames:
                imps[idx].append(join(args.input, scene, 'images', cam, frame))
                ccms[idx].append(cams[cam].ccm)

        imps = np.asarray(imps)
        ccms = np.asarray(ccms)
        pbar = tqdm(total=imps.size)
        for idx in range(len(basenames)):
            for i in range(0, len(imps[idx]), args.batch_size):
                imps_batch = imps[idx][i:i + args.batch_size]
                ccms_batch = ccms[idx][i:i + args.batch_size]
                outs_batch = np.asarray([imp.replace('images', 'images_calib') for imp in imps_batch])
                inps_batch = parallel_execution(list(imps_batch), action=load_unchanged_image)
                inps_batch = to_cuda(inps_batch)
                inps_batch = torch.stack(inps_batch)
                ccms_batch = to_cuda(ccms_batch)
                imgs_batch = calib_color(inps_batch, ccms_batch)[..., [2, 1, 0]].detach().cpu().numpy()
                parallel_execution(list(outs_batch), list(imgs_batch), action=partial(save_image, jpeg_quality=args.jpeg_quality))
                pbar.update(len(imps_batch))


if __name__ == '__main__':
    args = parse_args()
    main(args)
