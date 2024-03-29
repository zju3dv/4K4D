# 将单帧数据转换为Nerf格式
import os
import cv2
import math
import argparse
import numpy as np
from glob import glob

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


def convert_K(K, D, H, W):
    fl_x = K[0, 0]
    fl_y = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    k1, k2, p1, p2, p3 = D[0]

    angle_x = math.atan(W / (fl_x * 2)) * 2
    angle_y = math.atan(H / (fl_y * 2)) * 2
    # fovx = angle_x * 180 / math.pi
    # fovy = angle_y * 180 / math.pi

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "cx": cx,
        "cy": cy,
        "h": int(H),
        'w': int(W),
    }
    return out


def convertRT(RT0):
    RT = np.eye(4)
    RT[:3] = RT0
    c2w = np.linalg.inv(RT)
    c2w[0:3, 1] *= -1
    c2w[0:3, 2] *= -1  # flip the y and z axis
    # c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
    # c2w[2, :] *= -1  # flip whole world upside down
    return c2w


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/mnt/remote/D002/home/xuzhen/projects/large_gaussian/demo/cameras/00')
    parser.add_argument('--out_file', type=str, default='/mnt/remote/D002/home/xuzhen/projects/large_gaussian/demo/trajectories/path0.json')
    parser.add_argument('--images_dir', type=str, default='images')
    parser.add_argument('--img_ext', type=str, default='.png', help='Only used when not using a multi_frame_dataset for contructing file_path')
    parser.add_argument('--image_file', type=str, default='000000.jpg', help='Only used when --multi_frame_dataset for contructing file_path')
    parser.add_argument('--multi_frame_dataset', action='store_true')
    args = parser.parse_args()

    os.makedirs(dirname(args.out_file), exist_ok=True)
    cameras = read_camera(args.data_root)

    camera_names = list(cameras.keys())
    annots = {'frames': []}

    K = cameras[camera_names[0]]['K']
    D = cameras[camera_names[0]]['D']
    H = cameras[camera_names[0]]['H']
    W = cameras[camera_names[0]]['W']
    annots.update(convert_K(K, D, H, W))

    for camera_name in tqdm(camera_names):
        camera = cameras[camera_name]

        if not args.multi_frame_dataset:
            filename = f'{args.images_dir}/{camera_name}{args.img_ext}'
        else:
            filename = f'{args.images_dir}/{camera_name}/{args.image_file}'

        # convert RT
        c2w = convertRT(camera['RT'])
        info = {
            'file_path': filename,
            'transform_matrix': c2w.tolist(),
        }
        info.update(convert_K(camera['K'], camera['D'], camera['H'], camera['W']))
        annots['frames'].append(info)

    with open(args.out_file, 'w') as f:
        json.dump(annots, f, indent=4)

    log(yellow(f'Saved converted camera path to: {blue(args.out_file)}'))


if __name__ == '__main__':
    main()
