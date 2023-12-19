import numpy as np
import argparse, os, json
import imageio.v2 as imageio

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_root', type=str, default='')
    parser.add_argument('--easyvolcap_root', type=str, default='')
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    # clean and restart
    os.system(f'rm -rf {args.easyvolcap_root}')
    os.makedirs(args.easyvolcap_root, exist_ok=True)

    # global parameter
    global_count = 0
    easyvv_cameras = {}
    sh = imageio.imread(join(args.synthetic_root, 'train', 'r_0.png')).shape
    H, W = int(sh[0]), int(sh[1])


    def process_camera_image(args, split, cnt):
        # load the raw split information
        transforms = json.load(open(join(args.synthetic_root, f'transforms_{split}.json')))
        # get the number of images in the current split
        split_num = len(os.listdir(join(args.synthetic_root, f'{split}'))) if split != 'test' else len(os.listdir(join(args.synthetic_root, f'{split}'))) // 3
        for local_count in range(split_num):
            # create soft link for image
            img_sync_path = join(args.synthetic_root, f'{split}', f'r_{local_count}.{args.ext}')
            img_easy_path = join(args.easyvolcap_root, f'images', f'{cnt:03d}', f'00.{args.ext}')
            os.makedirs(join(args.easyvolcap_root, 'images', f'{cnt:03d}'), exist_ok=True)
            os.system(f'ln -s {img_sync_path} {img_easy_path}')
            # fetch and store camera parameters
            c2w_opengl = np.array(transforms['frames'][local_count]['transform_matrix']).astype(np.float32)
            c2w_opencv = c2w_opengl @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c_opencv = np.linalg.inv(c2w_opencv)
            easyvv_cameras[f'{cnt:03d}'] = {
                'R': w2c_opencv[:3, :3],
                'T': w2c_opencv[:3, 3:],
                'K': np.array([[0.5 * W / np.tan(0.5 * transforms['camera_angle_x']), 0, 0.5 * W],
                               [0, 0.5 * W / np.tan(0.5 * transforms['camera_angle_x']), 0.5 * H],
                               [0, 0, 1]]),
                'D': np.zeros((1, 5)),
            }
            # have a look at the first camera
            if cnt == 0: print(easyvv_cameras)
            cnt += 1
        return cnt

    # process one by one
    splits = ['train', 'val', 'test']
    for split in splits: global_count = process_camera_image(args, split, global_count)

    # write the cameras
    write_camera(easyvv_cameras, args.easyvolcap_root)
    log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, "{intri.yml,extri.yml}"))}'))
