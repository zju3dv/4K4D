# Converts raw dtu dataset format to easyvolcap
# Links images and maybe convert the whole folder

import os
import cv2
import argparse
import numpy as np
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import read_cam_file, read_pfm, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtu_root', type=str, default='/nas/home/xuzhen/datasets//dtu')
    parser.add_argument('--easyvolcap_root', type=str, default='data/dtu')
    parser.add_argument('--scale', type=float, default=200.0)  # rescale the cameras as in enerf
    args = parser.parse_args()

    scale = args.scale
    dtu_root = args.dtu_root
    easyvolcap_root = args.easyvolcap_root

    # Read all cameras
    num_cams = len(os.listdir(join(dtu_root, f'Cameras/train')))
    cams = dotdict()
    for i in range(num_cams):
        cam_path = join(dtu_root, f'Cameras/train/{i:08d}_cam.txt')
        ixt, ext, _ = read_cam_file(cam_path)  # w2c, c2w
        ext[:3, 3] = ext[:3, 3] / scale
        ixt[:2] = ixt[:2] * 4  # why?

        cam = dotdict()
        cam.K = ixt
        cam.R = ext[:3, :3]  # 3, 3
        cam.T = ext[:3, 3:]  # 3, 1
        cams[f'{i:06d}'] = cam  # seen as temporal camera

    def process_scene(scene):
        # Treat them as one single camera's input
        img_out_dir = join(easyvolcap_root, scene, 'images', '00')
        msk_out_dir = join(easyvolcap_root, scene, 'masks', '00')
        dpt_out_dir = join(easyvolcap_root, scene, 'depths', '00')
        cam_out_dir = join(easyvolcap_root, scene, 'cameras', '00')
        os.makedirs(img_out_dir, exist_ok=True)  # dtu/scanxxx
        os.makedirs(msk_out_dir, exist_ok=True)  # dtu/scanxxx
        os.makedirs(dpt_out_dir, exist_ok=True)  # dtu/scanxxx
        os.makedirs(cam_out_dir, exist_ok=True)  # dtu/scanxxx

        # Store camera parameters
        write_camera(cams, cam_out_dir)  # repeated training cameras
        log(yellow(f'Converted cameras saved to {blue(join(cam_out_dir, "{intri.yml,extri.yml}"))}'))

        def process_image(i):
            dpt_path = join(dtu_root, f'Depths_raw/{scene}/depth_map_{i:04d}.pfm')
            img_path = join(dtu_root, f'Rectified/{scene}_train/rect_{i+1:03d}_3_r5000.png')

            dpt = read_pfm(dpt_path)[0].astype(np.float32) / scale
            msk = (dpt > 0.).astype(np.uint8)
            # msk = np.ones_like(dpt) # treat all pixels as valid
            msk = cv2.resize(msk, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)[44:-44, 80:-80]
            dpt = cv2.resize(dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)[44:-44, 80:-80]

            # Writing and linking
            img_out_path = join(img_out_dir, f'{i:06d}.jpg')
            save_image(img_out_path, cv2.imread(img_path)[..., [2, 1, 0]], jpeg_quality=100)  # highest quality compression
            msk_out_path = join(msk_out_dir, f'{i:06d}.jpg')
            save_image(msk_out_path, msk[..., None] * 255)
            dpt_out_path = join(dpt_out_dir, f'{i:06d}.exr')
            save_image(dpt_out_path, dpt)

        parallel_execution(list(range(num_cams)), action=process_image)

    scenes = os.listdir(join(dtu_root, 'Rectified'))
    scenes = [s.replace('_train', '') for s in scenes]  # remove duplication
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
