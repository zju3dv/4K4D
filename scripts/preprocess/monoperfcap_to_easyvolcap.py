# Dataset conversion
# For now, only one static camera is supported
import os
import numpy as np
import argparse
from os.path import join, dirname
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera, write_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monoperfcap_root', default='/nas/home/xuzhen/datasets/monoperfcap_raw')
    parser.add_argument('--easyvolcap_root', default='/mnt/data3/home/xuzhen/datasets/monoperfcap')
    parser.add_argument('--humans', default=['Nadia_outdoor', 'Natalia_outdoor'], nargs='+')
    parser.add_argument('--camera_only', action='store_true')
    args = parser.parse_args()

    for human in args.humans:
        mono_dir = join(args.monoperfcap_root, human)
        easy_dir = join(args.easyvolcap_root, human)

        if not args.camera_only:
            image_path = join(mono_dir, human, 'images')
            out_image_dir = join(easy_dir, 'images/00')
            os.makedirs(out_image_dir, exist_ok=True)

            images = sorted(os.listdir(image_path))
            for i, img_file in enumerate(tqdm(images)):
                out_img_file = '{:06d}.jpg'.format(i)
                out_img_path = join(out_image_dir, out_img_file)
                in_img_path = join(image_path, img_file)
                os.system(f'cp {in_img_path} {out_img_path}')

        calib_path = join(mono_dir, 'calib.txt')
        out_calib_dir = easy_dir
        os.makedirs(out_calib_dir, exist_ok=True)
        content = open(calib_path).readlines()
        H, W = list(map(int, content[0].split()[1:]))
        ixt = np.asarray(list(map(float, content[2].split()[1:]))).reshape(4, 4)
        ext = np.asarray(list(map(float, content[3].split()[1:]))).reshape(4, 4)
        cam = dotdict()
        cam.H = H
        cam.W = W
        cam.K = ixt
        cam.R = ext[:3, :3]
        cam.T = ext[:3, 3:]
        write_camera({'00': cam}, out_calib_dir)
        log(yellow(f'Converted cameras saved to {blue(join(out_calib_dir, "{intri.yml,extri.yml}"))}'))


if __name__ == '__main__':
    main()
