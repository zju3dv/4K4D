import os
import cv2
import argparse
import numpy as np

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.colmap_utils import qvec2rotmat, read_model
# fmt: on


@catch_throw
def main(args):
    cameras, images, points3D = read_model(path=args.colmap)
    log(f"number of cameras: {len(cameras)}")
    log(f"number of images: {len(images)}")
    log(f"number of points3D: {len(points3D)}")

    intrinsics = {}
    for key in cameras.keys():
        p = cameras[key].params
        if cameras[key].model == 'SIMPLE_RADIAL':
            f, cx, cy, k = p
            K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[k, 0, 0, 0, 0]])
        elif cameras[key].model == 'PINHOLE':
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[0., 0., 0., 0., 0.]])
        else:  # OPENCV
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[p[4], p[5], p[6], p[7], 0.]])
        H, W = cameras[key].height, cameras[key].width
        intrinsics[key] = {'K': K, 'dist': dist, 'H': H, 'W': W}

    easycams = {}
    for key, val in sorted(images.items(), key=lambda item: item[0]):
        if args.sub in val.name:
            log(f'preparing camera: {val.name}(#{val.camera_id})')
            cam = intrinsics[val.camera_id].copy()
            t = val.tvec.reshape(3, 1)
            R = qvec2rotmat(val.qvec)
            cam['Rvec'] = cv2.Rodrigues(R)[0]
            cam['R'] = R
            cam['T'] = t * args.scale
            easycams[os.path.splitext(os.path.basename(val.name))[0]] = cam
        else:
            log(f'skipping camera: {val.name}(#{val.camera_id}) since {args.sub} not in {val.name}', 'yellow')

    # Dicts preserve insertion order in Python 3.7+. Same in CPython 3.6, but it's an implementation detail.
    easycams = dict(sorted(easycams.items(), key=lambda item: item[0]))
    write_camera(easycams, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/iphone/hdr_412')
    parser.add_argument('--colmap', type=str, default='colmap/colmap_sparse/0')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--sub', type=str, default='')  # only camera name containing this string will be saved
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    args.colmap = join(args.data_root, args.colmap)
    args.output = join(args.data_root, args.output)
    main(args)
