import os
import argparse
from os import listdir
from os.path import join
from easyvolcap.utils.console_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/mobile_stage/dance3')
    parser.add_argument('--bkgd_dir', default='bkgd')
    parser.add_argument('--bkgd_out', default='bkgd/images')
    parser.add_argument('--image', default='000000.jpg')
    parser.add_argument('--cam_ext', default='.jpg')
    parser.add_argument('--cam_fmt', default='{:02d}')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    cameras = sorted(listdir(join(args.data_root, args.bkgd_dir)))
    if args.backward:
        cameras = [c for c in cameras if not c.endswith(args.cam_ext)]
    else:
        cameras = [c[:-len(args.cam_ext)] for c in cameras if c.endswith(args.cam_ext)]
    cameras = list(map(int, cameras))

    if args.backward:
        run(f'mkdir -p {join(args.data_root, args.bkgd_out)}')
        for cam in cameras:
            cam_str = args.cam_fmt.format(cam)
            run(f'cp {join(args.data_root, args.bkgd_out, cam_str, args.image)} {join(args.data_root, args.bkgd_dir, cam_str + args.cam_ext)}')
    else:
        for cam in cameras:
            cam_str = args.cam_fmt.format(cam)
            run(f'mkdir -p {join(args.data_root, args.bkgd_out, cam_str)}')
            run(f'cp {join(args.data_root, args.bkgd_dir, cam_str + args.cam_ext)} {join(args.data_root, args.bkgd_out, cam_str, args.image)}')


if __name__ == '__main__':
    main()
