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
from easyvolcap.utils.data_utils import as_numpy_func, export_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--neural3dv_root', type=str, default='data/neural3dv')
    parser.add_argument('--easyvolcap_root', type=str, default='data/neural3dv')
    parser.add_argument('--camera_pose', type=str, default='poses_bounds.npy')
    parser.add_argument('--only', nargs='+', default=['sear_steak', 'coffee_martini', 'flame_steak'])
    args = parser.parse_args()

    scenes = os.listdir(args.neural3dv_root)
    scenes = [s for s in scenes if s in args.only]
    for scene in tqdm(scenes):
        # https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/datasets/llff.py#L177
        raw = np.load(join(args.neural3dv_root, scene, args.camera_pose), allow_pickle=True)  # 21, 17
        poses = raw[:, :15].reshape(-1, 3, 5)  # N, 3, 5
        bounds = raw[:, -2:]  # N, 2
        # Step 1: rescale focal length according to training resolution
        H, W, F = poses[0, :, -1]  # original intrinsics, same for all images

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right down front"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], poses[..., :1], -poses[..., 2:3], poses[..., 3:4]], -1)  # (N_images, 3, 4) exclude H, W, focal
        cameras = dotdict()

        names = sorted(os.listdir(join(args.neural3dv_root, scene, 'images')))
        for i in range(len(poses)):
            key = names[i]

            c2w = poses[i]
            w2c = as_numpy_func(affine_inverse)(c2w)

            cameras[key] = dotdict()
            cameras[key].R = w2c[:3, :3]
            cameras[key].T = w2c[:3, 3:]
            cameras[key].K = np.zeros_like(cameras[key].R)
            cameras[key].K[0, 0] = F
            cameras[key].K[1, 1] = F
            cameras[key].K[0, 2] = W / 2
            cameras[key].K[1, 2] = H / 2
            cameras[key].K[2, 2] = 1.0
            cameras[key].n = bounds[i, 0]  # camera has near and far
            cameras[key].f = bounds[i, 1]  # camera has near and far

        write_camera(cameras, join(args.easyvolcap_root, scene))
        log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, scene, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
