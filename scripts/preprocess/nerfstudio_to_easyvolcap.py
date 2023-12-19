import json
import argparse
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera


def opengl2opencv(c2w):
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    return c2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerfstudio_root', type=str, default='data/badminton/seq3/nerfstudio/camera_paths')
    parser.add_argument('--easyvolcap_root', type=str, default='data/paths/seq3')
    parser.add_argument('--json', type=str, default='2023-07-13_220007.json')
    args = parser.parse_args()
    data_path = join(args.nerfstudio_root, args.json)

    cameras = dotdict()

    jdata = json.load(open(data_path))
    H, W = jdata['render_height'], jdata['render_width']

    for i, pose in enumerate(jdata['camera_path']):
        cameras[f'{i:06d}'] = dotdict()
        fx = fy = 0.5 * H / np.tan(0.5 * pose['fov'] / 180.0 * np.pi)
        cameras[f'{i:06d}'].K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])
        c2w = opengl2opencv(np.array(pose['camera_to_world']).reshape((4, 4)))
        cameras[f'{i:06d}'].R = c2w[:3, :3].T
        cameras[f'{i:06d}'].T = -c2w[:3, :3].T @ c2w[:3, 3:]  # -R.T @ T, 3x3 @ 3x1 -> 3x1
        cameras[f'{i:06d}'].H = H
        cameras[f'{i:06d}'].W = W

    write_camera(cameras, args.easyvolcap_root)
    log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
