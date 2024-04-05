"""
Perform camera undistortion on easyvolcap dataset format
"""
from typing import Literal

import cv2
import kornia
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


def undistort(img: str, out: str, D: np.ndarray, backend: Literal['cv2', 'kornia', 'colmap']='cv2'):
    pass


@catch_throw
def main():
    args = dotdict(
        data_root='data/selfcap/0330_01',
        images_dir='images_dist',
        output_dir='images_undist',
        backend='cv2',  # cv2, colmap, kornia
        dist_opt_K=True,  # remove black edges
        cameras_dir='',  # root of data root of empty
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    camera_names = sorted(os.listdir(join(args.data_root, args.images_dir)))
    frames = sorted(os.listdir(join(args.data_root, args.images_dir, camera_names[0])))
    cameras = read_camera(join(args.data_root, args.cameras_dir))

    images = [join(args.data_root, args.images_dir, camera, frame) for camera in camera_names for frame in frames]
    D = [cameras[camera].D for camera in camera_names for frame in frames] # N, 5, 1
    breakpoint()


if __name__ == '__main__':
    main()
