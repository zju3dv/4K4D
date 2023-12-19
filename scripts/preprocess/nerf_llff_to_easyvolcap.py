import os
import argparse
import cv2
from os.path import join
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.colmap_utils import qvec2rotmat, read_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llff_root', type=str, default='')
    parser.add_argument('--easyvolcap_root', type=str, default='data/nerf_llff_data/fern')
    parser.add_argument('--ext', type=str, default='JPG')
    args = parser.parse_args()

    # clean and restart
    os.system(f'rm -rf {args.easyvolcap_root}')
    os.makedirs(args.easyvolcap_root, exist_ok=True)

    # read llff colmap model
    cameras, images, points3D = read_model(path=join(args.llff_root, 'sparse/0'))
    log(f"number of cameras: {len(cameras)}")
    log(f"number of images: {len(images)}")
    log(f"number of points3D: {len(points3D)}")

    intrinsics = {}
    for key in cameras.keys():
        p = cameras[key].params
        # convert the original intrinsic parameters
        if cameras[key].model == 'SIMPLE_RADIAL':   # SIMPLE_RADIAL
            f, cx, cy, k = p
            K = np.array([f, 0, cx, 0, f, cy, 0, 0, 1]).reshape(3, 3)
            dist = np.array([[k, 0, 0, 0, 0]])
        elif cameras[key].model == 'PINHOLE':       # PINHOLE
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[0., 0., 0., 0., 0.]])
        else:                                       # OPENCV
            K = np.array([[p[0], 0, p[2], 0, p[1], p[3], 0, 0, 1]]).reshape(3, 3)
            dist = np.array([[p[4], p[5], p[6], p[7], 0.]])
        H, W = cameras[key].height, cameras[key].width
        intrinsics[key] = {'K': K, 'dist': dist, 'H': H, 'W': W}

    easycams = {}
    llff_image_root = join(args.llff_root, 'images')
    easy_image_root = join(args.easyvolcap_root, 'images')
    for key, val in sorted(images.items(), key=lambda item: item[0]):
        log(f'preparing camera: {val.name}(#{val.camera_id})')
        # key and val.camera_id both start from 1
        cam = intrinsics[val.camera_id].copy()
        # transform the rotatoin and translation matrices
        t = val.tvec.reshape(3, 1)
        R = qvec2rotmat(val.qvec)
        cam['Rvec'] = cv2.Rodrigues(R)[0]
        cam['R'] = R
        cam['T'] = t
        # llff dataset has only one camera, namely monocular
        easycams[f'{(key-1):03d}'] = cam

        log(f'preparing images: {val.name}')
        llff_image_path = join(llff_image_root, val.name)
        easy_image_path = join(easy_image_root, f'{(key-1):03d}', f'00.png')
        os.makedirs(os.path.dirname(easy_image_path), exist_ok=True)
        os.system(f'ln -s {llff_image_path} {easy_image_path}')

    # Dicts preserve insertion order in Python 3.7+. Same in CPython 3.6, but it's an implementation detail.
    easycams = dict(sorted(easycams.items(), key=lambda item: item[0]))
    write_camera(easycams, args.easyvolcap_root)
    log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, "{intri.yml,extri.yml}"))}'))
