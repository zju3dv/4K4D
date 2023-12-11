# This script is used to compute the view-wise depth range for all the cameras in the dataset from
# an existing point cloud. The result will be saved in the `n` and `f` entry in `intri.yml`.
# Using this script requires easyvolcap or easymocap format data (i.e. intri.yml and extri.yml).

import argparse
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts
from easyvolcap.utils.easy_utils import read_camera, write_camera


def compute_range(pts, w2c, K, H, W):
    # Project the point cloud to the image plane
    pts = pts @ w2c[:3, :3].T + w2c[:3, 3].T  # (N, 3)
    pts = pts @ K.T  # (N, 3)
    # Filter out the points outside the image plane
    uv = pts[:, :2] / pts[:, 2:]  # (N, 2)
    uv_msk = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)  # (N,)
    uv_msk = np.logical_and(uv_msk, pts[:, 2] > 0)  # (N,)
    # Percentile the depth range for the points inside the image plane
    z_vals = pts[..., 2][uv_msk]  # (N,)
    return np.percentile(z_vals, 0.1), np.percentile(z_vals, 99.9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, default='data/enerf_outdoor/actor1_4/optimized')
    parser.add_argument('--output_root', type=str, default='data/enerf_outdoor/actor1_4/optimized/ranged')
    parser.add_argument('--pts_path', type=str, default='')
    parser.add_argument('--convention', choices=['opencv', 'opengl'], default='opencv',
                        help='choose the camera coordinate convention (opencv or opengl)')
    parser.add_argument('--n_dilate_ratio', type=float, default=0.95)
    parser.add_argument('--f_dilate_ratio', type=float, default=1.05)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--width', type=int, default=1920)
    args = parser.parse_args()

    # Read in the original camera parameters
    input_root = args.input_root
    cameras = read_camera(join(input_root, 'intri.yml'), join(input_root, 'extri.yml'))
    camera_names = np.asarray(sorted(list(cameras.keys())))

    # Read in the point cloud
    pts_path = args.pts_path
    pts, _, _, _ = load_pts(pts_path)  # (N, 3)

    # Convert the point cloud to OpenCV camera coordinate if needed
    if args.convention == 'opengl':
        pts[:,  2] = pts[:,  2] * -1
        pts[:, :2] = pts[:, [1, 0]]

    out_cameras = dict()
    # Project the point cloud to every view and compute the depth range
    for name in camera_names:
        cam = cameras[name].copy()
        w2c = np.concatenate([np.asarray(cam.R), np.asarray(cam.T)], axis=1)  # (3, 4)
        n, f = compute_range(pts, w2c, np.asarray(cam.K), args.height, args.width)
        cam.n = n * args.n_dilate_ratio
        cam.f = f * args.f_dilate_ratio
        out_cameras[name] = cam

    # Save the result
    run(f'mkdir -p {args.output_root}')
    write_camera(out_cameras, args.output_root)


if __name__ == '__main__':
    main()
