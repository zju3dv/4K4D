# this script will convert DynaCap format data to easymocap format
# .avi img -> images directory
# .avi fg -> green_screen mask directory
# .calibration -> extri.yml and intri.yml (Skeletool Camera Calibration File V1.0 -> OpenCV format)
# background -> background images merged together

import torch
from tqdm import tqdm
from typing import List
import os
import argparse
import numpy as np
from os.path import join
from termcolor import cprint, colored

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import run
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, list_to_numpy, list_to_tensor, save_image
# fmt: on


def skeletool_to_easymocap(calibration_path: str, easymocap_dir: str):
    """
    Convert Skeletool camera calibration file to easymocap format
    :param calibration_path: path to Skeletool camera calibration file
    :param easymocap_dir: where to store intri.yml and extri.yml
    :return:
    """

    def process_intrinsic(d, n, *arg):
        d[n]['K'] = np.array(list(map(float, arg))).reshape(4, 4)[:3, :3]  # row major

    def process_extrinsic(d, n, *arg):
        RT = np.array(list(map(float, arg))).reshape(4, 4)  # row major
        d[n]['R'] = RT[:3, :3]  # 3, 3
        d[n]['T'] = RT[:3, 3:] / 1000  # 3, 1, wtf? is this some strange skeletool convension?

    cameras = {}
    processor = {
        'sensor': lambda d, n, *arg: None,
        'size': lambda d, n, *arg: None,
        'animated': lambda d, n, *arg: None,
        'intrinsic': process_intrinsic,
        'extrinsic': process_extrinsic,
        'radial': lambda d, n, *arg: None,
        'skeletool': lambda d, n, *arg: None,
    }
    id_key = 'name'
    name = None
    with open(calibration_path, "r") as f:
        for line in f:
            line = line.strip().split()
            key = line[0].lower()
            arg = line[1:]
            if key == id_key:
                name = f'{int(arg[0]):02d}'
                cameras[name] = {}  # create the camera
            else:
                processor[key](cameras, name, *arg)

    write_camera(cameras, easymocap_dir)
    return cameras


def video_to_frames(video_path: str, frames_dir: str, ext='.jpg'):
    """
    Convert video to frames
    :param video_path: path to video
    :param frames_dir: where to store frames
    :return:
    """

    os.makedirs(frames_dir, exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-start_number', '0',
        '-q:v', '2',
        f'{frames_dir}/%06d{ext}'
    ]
    run(' '.join(command))


def inference_segmentation(data_root: str):
    run(f'python scripts/segmentation/inference_schp.py --data_root {data_root}')
    run(f'python scripts/segmentation/inference_robust_video_matting.py --data_root {data_root}')
    run(f'python scripts/segmentation/inference_bkgdmattev2.py --data_root {data_root}')


def merge_background(src_dir: str, tar_dir: str):
    # this script will load all images inside one specific background folder
    # and merge them into one (taking the mean)
    # then convert to uint8 properly: uint8(x+0.5)
    # note that camera names will be infered from the directory names: int(x):02d.jpg
    cam_dirs = os.listdir(src_dir)  # will not be sorted good enough
    cam_names = [f'{int(c):02d}' for c in cam_dirs]
    for cam, cam_dir in tqdm(zip(cam_names, cam_dirs), total=len(cam_names)):
        cam_path = join(src_dir, cam_dir)
        img_names = os.listdir(cam_path)
        img_paths = [join(cam_path, i) for i in img_names]

        img = list_to_tensor(parallel_execution(img_paths, action=load_image), 'cpu')

        img = img.mean(dim=0)
        img = img.permute(1, 2, 0).detach().cpu().numpy()

        save_image(join(tar_dir, f'{cam}.jpg'), img)


def main():
    parser = argparse.ArgumentParser('helper script to convert DynaCap format data to easymocap format')
    parser.add_argument('--dynacap_root', type=str, default='/nas/home/xuzhen/datasets/dynacap-original/Vlad', help='path to DynaCap directory')
    parser.add_argument('--easymocap_root', type=str, default='/nas/home/xuzhen/datasets/dynacap/vlad', help='path to easymocap directory')
    parser.add_argument('--calib_file', type=str, default='cameras.calibration')
    parser.add_argument('--video_dir', type=str, default='training/img')
    parser.add_argument('--green_dir', type=str, default='training/fg')
    parser.add_argument('--video_prefix', type=str, default='c_')
    parser.add_argument('--video_postfix', type=str, default='.avi', help='including file extension')
    args = parser.parse_args()

    image_dir = 'images'
    green_dir = 'green'

    cameras = skeletool_to_easymocap(join(args.dynacap_root, args.calib_file), args.easymocap_root)  # will write intri.yml and extri.yml to args.easymocap_root

    def extract_image(cam): video_to_frames(join(args.dynacap_root, args.video_dir, args.video_prefix+cam+args.video_postfix), join(args.easymocap_root, image_dir, cam), '.jpg')
    def extract_green(cam): video_to_frames(join(args.dynacap_root, args.green_dir, args.video_prefix+cam+args.video_postfix), join(args.easymocap_root, green_dir, cam), '.jpg')

    parallel_execution(list(cameras.keys()), action=extract_image, num_workers=32)

    parallel_execution(list(cameras.keys()), action=extract_green, num_workers=32)

    merge_background(join(args.dynacap_root, 'background'), join(args.easymocap_root, 'bkgd'))

    inference_segmentation(args.easymocap_root)


if __name__ == '__main__':
    main()

"""
python scripts/dynacap/dynacap_to_easymocap.py --dynacap_root /nas/home/xuzhen/datasets/dynacap-original/Vlad --easymocap_root /nas/home/xuzhen/datasets/dynacap/vlad
python scripts/dynacap/dynacap_to_easymocap.py --dynacap_root /nas/home/xuzhen/datasets/dynacap-original/OlekDesert --easymocap_root /nas/home/xuzhen/datasets/dynacap/olek
python scripts/dynacap/dynacap_to_easymocap.py --dynacap_root /nas/home/xuzhen/datasets/dynacap-original/FranziRed --easymocap_root /nas/home/xuzhen/datasets/dynacap/franzi
"""
