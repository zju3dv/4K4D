import os
import argparse
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/bullet/final')
    parser.add_argument('--dirs', default=['images', 'videos', 'masks', 'cameras'])
    args = parser.parse_args()

    for dir in args.dirs:
        if not exists(join(args.data_root, dir)): continue
        for idx, cam in enumerate(sorted(os.listdir(join(args.data_root, dir)))):
            _, ext = splitext(cam)
            new_name = f"{idx:02d}{ext}"
            if new_name != cam:
                os.system(f'mv {join(args.data_root, dir, cam)} {join(args.data_root, dir, new_name)}')


if __name__ == '__main__':
    main()
