import os
import cv2
import json
import argparse
import operator
import numpy as np
import imageio.v2 as imageio

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.parallel_utils import parallel_execution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnerf_root', type=str, default='')
    parser.add_argument('--easyvolcap_root', type=str, default='data/dnerf')
    parser.add_argument('--black_bkgds', action='store_true')
    args = parser.parse_args()

    dnerf_root = args.dnerf_root
    easyvolcap_root = args.easyvolcap_root

    def process_camera_image(dnerf_path, easyvolcap_path, split, frames, camera_angle_x, H, W):
        # Define and create output image path and mask path
        img_out_dir = join(easyvolcap_path, split, f'images', '00')
        msk_out_dir = join(easyvolcap_path, split, f'masks', '00')
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(msk_out_dir, exist_ok=True)

        cameras = dotdict()
        # Remove frames with the same timestamp
        for cnt, frame in enumerate(frames):
            # Create soft link for image
            img_dnerf_path = join(dnerf_path, frame['file_path'][2:] + '.png')
            img_easyvolcap_path = join(img_out_dir, f'{cnt:03d}.png')
            img = imageio.imread(img_dnerf_path) / 255.0
            if args.black_bkgds: img = img[..., :3]
            else: img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            imageio.imwrite(img_easyvolcap_path, (img * 255.0).astype(np.uint8))

            # Create mask for the image
            msk = imageio.imread(img_dnerf_path).sum(axis=-1) > 0
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
        dnerf_path = join(dnerf_root, scene)
        easyvolcap_path = join(easyvolcap_root, scene)

        sh = imageio.imread(join(dnerf_path, 'train', sorted(os.listdir(join(dnerf_path, 'train')))[1])).shape
        H, W = int(sh[0]), int(sh[1])

        # Load frames information of all splits
        splits = ['train', 'val', 'test']
        for split in splits:
            # Load all frames information
            frames = json.load(open(join(dnerf_path, f'transforms_{split}.json')))['frames']
            frames = sorted(frames, key=operator.itemgetter('time'))
            camera_angle_x = json.load(open(join(dnerf_path, f'transforms_{split}.json')))['camera_angle_x']
            # Process camera parameters
            cameras = process_camera_image(dnerf_path, easyvolcap_path, split, frames, camera_angle_x, H, W)
            # Write camera parameters, treat dnerf dataset as one camera monocular video dataset
            write_camera(cameras, join(easyvolcap_path, split, 'cameras', '00'))
            log(yellow(f'Converted cameras saved to {blue(join(easyvolcap_path, split, "cameras", "00", "{intri.yml,extri.yml}"))}'))

    # Clean and restart
    os.system(f'rm -rf {easyvolcap_root}')
    os.makedirs(easyvolcap_root, exist_ok=True)

    # Convert all scenes
    scenes = os.listdir(dnerf_root)
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
