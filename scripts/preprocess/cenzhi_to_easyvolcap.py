import pickle
import argparse
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.data_utils import save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cenzhi_root', type=str, default='/nas/home/cenzhi/nerf_data/t_pose')
    parser.add_argument('--easyvolcap_root', type=str, default='data/gaussian/tpose360')
    parser.add_argument('--pickle', type=str, default='debug_360cam.pkl')
    args = parser.parse_args()
    data_root = join(args.cenzhi_root, args.pickle)

    cameras = dotdict()

    with open(data_root, 'rb') as f:
        pdata = pickle.load(f)
        heatmap, poses = pdata['heatmap'], pdata['poses']  # camera 2 world pose
        # render_heatmap, render_poses = pdata['render_heatmap'], pdata['render_poses']
        hwf, i_split = pdata['hwf'], pdata['i_split']
        H, W, focal = hwf

    for i, pose in enumerate(poses):
        cameras[f'{i:06d}'] = dotdict()
        cameras[f'{i:06d}'].K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
        cameras[f'{i:06d}'].R = pose[:3, :3].T
        cameras[f'{i:06d}'].T = -pose[:3, :3].T @ pose[:3, 3:]  # -R.T @ T, 3x3 @ 3x1 -> 3x1

    run(f'mkdir -p {args.easyvolcap_root}/images/00')
    run(f'mkdir -p {args.easyvolcap_root}/masks/00')
    write_camera(cameras, join(args.easyvolcap_root, 'cameras', '00'))
    log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, split, "cameras", "00", "{intri.yml,extri.yml}"))}'))

    for i, heat in enumerate(heatmap):
        heat = heat.sum(axis=-1, keepdims=True).repeat(3, axis=-1)
        save_image(join(args.easyvolcap_root, 'images', '00', f'{i:06d}.png'), heat)
        save_image(join(args.easyvolcap_root, 'mask', '00', f'{i:06d}.png'), heat[..., :1] > 0.0001)


if __name__ == '__main__':
    main()
