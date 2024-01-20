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
    parser.add_argument('--nerf_root', type=str, default='/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital/4')
    parser.add_argument('--volcap_root', type=str, default='/mnt/data/home/shuaiqing/Code/MultiNBFast/cache/Hospital/4')
    parser.add_argument('--transforms_file', type=str, default='transforms_train.json')
    parser.add_argument('--intri_file', type=str, default='intri.yml')
    parser.add_argument('--extri_file', type=str, default='extri.yml')
    args = parser.parse_args()

    # clean and restart
    os.makedirs(args.volcap_root, exist_ok=True)

    # load the raw split information
    transforms = dotdict(json.load(open(join(args.nerf_root, args.transforms_file))))
    H, W = transforms.h, transforms.w

    # global parameter
    evc_cams = {}

    # get the number of images in the current split
    for local_count in range(len(transforms.frames)):
        # fetch and store camera parameters
        c2w_opengl = np.array(transforms.frames[local_count].transform_matrix).astype(np.float32)
        c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        w2c_opencv = np.linalg.inv(c2w_opencv)
        if 'fl_x' in transforms.frames[local_count]:
            fx, fy, cx, cy = transforms.frames[local_count].fl_x, transforms.frames[local_count].fl_y, \
                transforms.frames[local_count].cx, transforms.frames[local_count].cy
            H, W = transforms.frames[local_count].h, transforms.frames[local_count].w
        else:
            fx, fy, cx, cy = 0.5 * W / np.tan(0.5 * transforms.camera_angle_x), \
                0.5 * W / np.tan(0.5 * transforms.camera_angle_x), 0.5 * W, 0.5 * H

        evc_cams[f'{local_count:06d}'] = {
            'R': w2c_opencv[:3, :3],
            'T': w2c_opencv[:3, 3:],
            'K': np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0, 0, 1]]),
            'D': np.zeros((1, 5)),
            'H': H,
            'W': W,
        }

    # write the cameras
    write_camera(
        evc_cams,
        args.volcap_root,
        join(args.volcap_root, args.intri_file),
        join(args.volcap_root, args.extri_file)
    )
    log(yellow(f'Converted cameras saved to {blue(join(args.volcap_root, f"{{{args.intri_file},{args.extri_file}}}"))}'))


if __name__ == '__main__':
    main()
