# Convert neural3dv dataset to easyvolcap format
# Assume we've got the images processed by haotong
# Need some validation since we're kind of blindly converting from pose_bounds.npy
# Need a way to store the bound info in the camera parameters file

import argparse
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.data_utils import as_numpy_func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--htcode_root', type=str, default='/mnt/data/home/linhaotong/Codes/K-Planes/data')
    parser.add_argument('--easyvolcap_root', type=str, default='data/neural3dv/cut_roasted_beef')
    parser.add_argument('--camera_pose', type=str, default='poses.npy')
    args = parser.parse_args()

    pose = np.load(join(args.htcode_root, args.camera_pose), allow_pickle=True).item()  # 20, 4, 4
    c2w = as_numpy_func(affine_inverse)(pose['exts'])  # 20, 4, 4

    # Undo ht conversion
    c2w[:, 0:3, 1:3] *= -1
    c2w = c2w[:, [1, 0, 2, 3], :]
    c2w[:, 2, :] *= -1
    # breakpoint()

    # # From OpenGL to OpenCV
    # c2w[:, 0] *= 1  # flip y
    # c2w[:, 1, :3] *= -1  # flip y
    # c2w[:, 2] *= -1  # flip z
    # c2w[:, 0, :] *= -1
    c2w[:, 2, :] *= -1
    c2w[:, :3, [1, 2]] *= -1
    # c2w[:, [1, 2], :3] *= -1
    # breakpoint()

    # c2w[:, :3, 0], c2w[:, :3, 1] = c2w[:, :3, 1], c2w[:, :3, 0]  # flip x y rotation
    # c2w[:, 0, 3], c2w[:, 1, 3] = c2w[:, 1, 3], c2w[:, 0, 3]  # flip x y translation
    # c2w[:, 1, 3] *= -1  # invert y translation
    # c2w[:, :3, 0] *= -1  # invert x rotation
    w2c = as_numpy_func(affine_inverse)(c2w)  # 20, 4, 4
    # w2c = pose['exts']
    # w2c = w2c[:, [1, 0, 2, 3], :]
    ixt = pose['ixt']

    names = sorted(os.listdir(join(args.easyvolcap_root, 'images')))
    cameras = dotdict()
    for name, ext in zip(names, w2c):
        R = ext[:3, :3]
        T = ext[:3, 3:]
        K = ixt.copy()
        K[:2] *= 2
        cameras[name] = dotdict()
        cameras[name].K = K
        cameras[name].R = R
        cameras[name].T = T

    write_camera(cameras, args.easyvolcap_root)
    log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, "{intri.yml,extri.yml}"))}'))

if __name__ == '__main__':
    main()
