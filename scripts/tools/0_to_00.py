import os
import argparse
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='data/mobile_stage/female')
    args = parser.parse_args()

    for cam in os.listdir(join(args.input, 'images')):
        new_name = f"{int(cam):02d}"
        if new_name != cam:
            os.system(f'mv {join(args.input, "images", cam)} {join(args.input, "images", new_name)}')

    cams = read_camera(join(args.input, 'intri.yml'), join(args.input, 'extri.yml'))
    new_cams = {}
    for cam in cams:
        new_cams[f"{int(cam):02d}"] = cams[cam]
    write_camera(new_cams, args.input)


if __name__ == '__main__':
    main()
