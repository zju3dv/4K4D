import numpy as np
import argparse
import os
import json
import imageio.v2 as imageio

from PIL import Image
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_root', type=str, default='/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital')
    parser.add_argument('--volcap_root', type=str, default='/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital')
    parser.add_argument('--transform_file', type=str, default='train_100.json')
    args = parser.parse_args()

    # clean and restart
    os.makedirs(args.volcap_root, exist_ok=True)

    # load the raw split information
    transforms = dotdict(json.load(open(join(args.nerf_root, args.transform_file))))
    H, W = transforms.h, transforms.w

    # global parameter
    evc_cams = {}

    # get the number of images in the current split
    for local_count in range(len(transforms.frames)):
        # fetch and store camera parameters
        c2w_opengl = np.array(transforms.frames[local_count].transform_matrix).astype(np.float32)
        c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        w2c_opencv = np.linalg.inv(c2w_opencv)
        evc_cams[f'{local_count:06d}'] = {
            'R': w2c_opencv[:3, :3],
            'T': w2c_opencv[:3, 3:],
            'K': np.array([[0.5 * W / np.tan(0.5 * transforms.camera_angle_x), 0, 0.5 * W],
                           [0, 0.5 * W / np.tan(0.5 * transforms.camera_angle_x), 0.5 * H],
                           [0, 0, 1]]),
            'D': np.zeros((1, 5)),
        }

    # write the cameras
    write_camera(evc_cams, args.volcap_root)
    log(yellow(f'Converted cameras saved to {blue(join(args.volcap_root, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
