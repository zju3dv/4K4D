import os
import argparse
import numpy as np
from glob import glob
from os.path import join

# fmt: off
import sys
sys.path.append('.')

from easyvolcap.utils.console_utils import run, log
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, save_image, variance_of_laplacian
# fmt: on


def load_and_compute_sharpness(img_path: str):
    img = load_image(img_path)
    sharpness = variance_of_laplacian(img)
    return sharpness


def discard_blurry(img_path: str, value: float, mean: float, threshold: float):
    if value < mean * threshold:
        log(f'discarding: {img_path} due to {value} < {mean*threshold}')
        os.remove(img_path)


def rearrange_dir(dir_path: str):
    images = glob(join(dir_path, '*.jpg')) + glob(join(dir_path, '*.png'))  # but they should not coexist
    images = sorted(images)
    log(f'number of images: {len(images)}')
    for i in range(len(images)):
        os.rename(images[i], join(dir_path, f'{i:06d}.jpg'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='data/iphone/dark_412/images/00')
    parser.add_argument("--threshold", default=0.25, type=float)
    args = parser.parse_args()

    log('loading images and computing sharpness')
    images = glob(join(args.data_root, '*.jpg')) + glob(join(args.data_root, '*.png'))  # but they should not coexist
    images = sorted(images)
    log(f'original number of images: {len(images)}')
    sharpness = parallel_execution(images, action=load_and_compute_sharpness, print_progress=True)
    mean = np.mean(sharpness)
    log(f'mean sharpness of loaded images: {mean}, discarding threshold: {mean * args.threshold}')

    log('discarding images')
    parallel_execution(images, sharpness, mean, args.threshold, action=discard_blurry)

    log('rearranging directory')
    rearrange_dir(args.data_root)


if __name__ == '__main__':
    main()
