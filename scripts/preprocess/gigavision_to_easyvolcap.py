# Converts raw genebody dataset format to easyvolcap
# Links images and maybe convert the whole folder

import os
import cv2
import argparse
import numpy as np
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gigavision_root', type=str, default='/nas/home/shuaiqing/share/Giga')
    parser.add_argument('--easyvolcap_root', type=str, default='./data/gigavision')
    args = parser.parse_args()

    gigavision_root = args.gigavision_root
    easyvolcap_root = args.easyvolcap_root

    def process_scene(scene: str):
        giga_dir = join(gigavision_root, scene)
        easy_dir = join(easyvolcap_root, scene)

        # Move / copy images
        os.makedirs(join(easy_dir, 'images', '00'), exist_ok=True)
        def process_image(img_path: str):
            giga_img_path = join(giga_dir, 'images', img_path)
            easy_img_path = join(easy_dir, 'images', '00', img_path)
            run(f'cp {giga_img_path} {easy_img_path}', quite=True)  # might be moving from nas to local, use copy instead of ln
        parallel_execution(sorted(os.listdir(join(giga_dir, 'images'))), action=process_image)

        # Parse cameras
        def process_cam(cam_path: str):
            giga_cam_path = join(giga_dir, 'cams', cam_path)
            with open(giga_cam_path, 'r') as f:
                lines = f.readlines()
            ext0 = lines[1].strip().split()
            ext1 = lines[2].strip().split()
            ext2 = lines[3].strip().split()
            ext3 = lines[4].strip().split()
            ext = np.array([ext0, ext1, ext2, ext3], dtype=np.float32)

            ixt0 = lines[7].strip().split()
            ixt1 = lines[8].strip().split()
            ixt2 = lines[9].strip().split()
            ixt = np.array([ixt0, ixt1, ixt2], dtype=np.float32)

            R = ext[:3, :3]
            T = ext[:3, :1]
            K = ixt
            cam = dotdict()
            cam.R = R
            cam.T = T
            cam.K = K

            key = cam_path.split('_')[0]
            return key, cam
        keys, cams = zip(*parallel_execution(os.listdir(join(giga_dir, 'cams')), action=process_cam))
        cams = {k: c for k, c in zip(keys, cams)}
        cam_dir = join(easy_dir, 'cameras', '00')
        os.makedirs(cam_dir, exist_ok=True)
        write_camera(cams, cam_dir)
        log(yellow(f'Converted cameras saved to {blue(join(cam_dir, "{intri.yml,extri.yml}"))}'))

    scenes = os.listdir(gigavision_root)
    scenes = sorted(scenes)
    parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)


if __name__ == '__main__':
    main()
