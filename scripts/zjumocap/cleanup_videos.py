import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('--transpose_image', action='store_true')
parser.add_argument('--cleanup_camera_name', action='store_true')
parser.add_argument('--cleanup_image_name', action='store_true')
parser.add_argument('--cleanup_annot_name', action='store_true')
args = parser.parse_args()

import cv2
import os
import sys
sys.path.append('.')
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.console_utils import log, run

from os.path import join
from glob import glob
from tqdm import tqdm

images = join(args.input, 'images')
if args.cleanup_camera_name:
    cameras = os.listdir(images) # only 1 level inside
    for cam in cameras:
        old_cam = join(images, cam)
        new_cam = join(images, cam[:2])
        if old_cam != new_cam:
            run(f'mv {old_cam} {new_cam}', quite=True)

if args.transpose_image:
    all_files = glob(join(images, '**', '*.jpg')) # glob all images
    def transpose(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(img_path, img[::-1, ::-1])
    parallel_execution(all_files, action=transpose, num_workers=64, print_progress=True)

if args.cleanup_image_name:
    # find smallest image name, clean them up with this value
    image_list = [int(os.path.splitext(f)[0]) for f in os.listdir(join(images, '00'))]
    ext = os.path.splitext(os.listdir(join(images, '00'))[0])[-1]
    smallest = min(image_list)
    cameras = os.listdir(images)
    for cam in cameras:
        cam_path = join(images, cam)
        image_list = list(range(len(os.listdir(cam_path))))
        for new in tqdm(image_list):
            old = new + smallest
            new_name = join(cam_path, f"{new:06d}{ext}")
            old_name = join(cam_path, f"{old:06d}{ext}")
            if new_name != old_name:
                run(f'mv {old_name} {new_name}', quite=True, dry_run=False)
