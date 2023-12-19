# Converts raw dnerf synthetic dataset format to easyvolcap
# Links images and maybe convert the whole folder

from easyvolcap.utils.data_utils import read_cam_file, read_pfm
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
import os
import cv2
import json
import argparse
import operator
import numpy as np
import imageio.v2 as imageio
from os.path import join, exists

import sys
sys.path.append('.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnerf_root', type=str, default='')
    parser.add_argument('--easyvolcap_root', type=str, default='data/dnerf')
    args = parser.parse_args()

    dnerf_root = args.dnerf_root
    easyvolcap_root = args.easyvolcap_root

    def process_camera_image(dnerfs_path, easyvv_path, split, frames, camera_angle_x, H, W):
        # Define and create output image path and mask path
        img_out_dir = join(easyvv_path, split, f'images', '00')
        msk_out_dir = join(easyvv_path, split, f'masks', '00')
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(msk_out_dir, exist_ok=True)

        cameras = dotdict()
        # Remove frames with the same timestamp
        for cnt, frame in enumerate(frames):
            # Create soft link for image
            img_dnerfs_path = join(dnerfs_path, frame['file_path'][2:] + '.png')
            img_easyvv_path = join(img_out_dir, f'{cnt:03d}.png')
            os.system(f'ln -s {img_dnerfs_path} {img_easyvv_path}')

            # Create mask for the image
            msk = imageio.imread(img_dnerfs_path).sum(axis=-1) > 0
            msk = msk.astype(np.uint8) * 255
            cv2.imwrite(join(msk_out_dir, f'{cnt:03d}.png'), msk)

            # Fetch and store camera parameters
            c2w_opengl = np.array(frame['transform_matrix']).astype(np.float32)
            c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c_opencv = np.linalg.inv(c2w_opencv)
            cameras[f'{cnt:03d}'] = {
                'R': w2c_opencv[:3, :3],
                'T': w2c_opencv[:3, 3:],
                'K': np.array([[0.5 * W / np.tan(0.5 * camera_angle_x), 0, 0.5 * W],
                               [0, 0.5 * W / np.tan(0.5 * camera_angle_x), 0.5 * H],
                               [0, 0, 1]]),
                'D': np.zeros((1, 5)),
                't': np.array(frame['time']).astype(np.float32),
                'H': H, 'W': W,
            }
            cnt += 1

        return cameras

    def process_scene(scene):
        # Create soft link for scene
        dnerfs_path = join(dnerf_root, scene)
        easyvv_path = join(easyvolcap_root, scene)

        sh = imageio.imread(join(dnerfs_path, 'train', sorted(os.listdir(join(dnerfs_path, 'train')))[1])).shape
        H, W = int(sh[0]), int(sh[1])

        # Load frames information of all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            # Load all frames information
            frames = json.load(open(join(dnerfs_path, f'transforms_{split}.json')))['frames']
            frames = sorted(frames, key=operator.itemgetter('time'))
            camera_angle_x = json.load(open(join(dnerfs_path, f'transforms_{split}.json')))['camera_angle_x']
            # Process camera parameters
            cameras = process_camera_image(dnerfs_path, easyvv_path, split, frames, camera_angle_x, H, W)
            # Write camera parameters, treat dnerf dataset as one camera monocular video dataset
            write_camera(cameras, join(easyvv_path, split, 'cameras', '00'))
            log(yellow(f'Converted cameras saved to {blue(join(easyvv_path, split, "cameras", "00", "{intri.yml,extri.yml}"))}'))

    # Clean and restart
    os.system(f'rm -rf {easyvolcap_root}')
    os.makedirs(easyvolcap_root, exist_ok=True)

    # Convert all scenes
    scenes = os.listdir(dnerf_root)
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
