"""
Convert neural3dv dataset to easyvolcap format
Assume we've got the images processed by haotong
Need some validation since we're kind of blindly converting from pose_bounds.npy
Need a way to store the bound info in the camera parameters file

Assume ./data/neural3dv/XXX
in easyvolcap:
python3 scripts/preprocess/neural3dv_to_easyvolcap.py --only XXX
python3 scripts/colmap/easymocap_to_colmap.py --data_root data/neural3dv/XXX --image_dir images --output_dir colmap
in spg_colmap:
python3 sfm_renbody.py --root_dir ./data/neural3dv/XXX/colmap --colmap_path $PATHTOCOLMAP
"""

import argparse
import subprocess
import numpy as np
from glob import glob
import subprocess
from pathlib import Path
from tqdm import tqdm

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import write_camera
from easyvolcap.utils.math_utils import affine_inverse
from easyvolcap.utils.data_utils import as_numpy_func, export_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural3dv_root', type=str, default='data/neural3dv')
    parser.add_argument('--easyvolcap_root', type=str, default='data/neural3dv')
    parser.add_argument('--camera_pose', type=str, default='poses_bounds.npy')
    parser.add_argument('--only', nargs='+', default=['sear_steak', 'cook_spinach', 'coffee_martini', 'flame_steak', 'flame_salmon', 'cut_roasted_beef'])  # NOTE: do not add cut_roasted_beef for 4k4d
    args = parser.parse_args()

    scenes = os.listdir(args.neural3dv_root)
    if len(args.only): scenes = [s for s in scenes if s in args.only]
    # for scene in tqdm(scenes):
    for scene in scenes:
        videos = sorted(glob(join(args.neural3dv_root, scene, 'videos', '*.mp4'), recursive=True))
        for v in videos:
            dirname = basename(v).split('.')[0][-2:]
            if not exists(join(args.easyvolcap_root, scene, 'images', dirname)):
                os.makedirs(join(args.easyvolcap_root, scene, 'images', dirname), exist_ok=True)
                cmd = [
                    'ffmpeg',
                    '-i', v,
                    '-vf', 'fps=30',
                    '-q:v', '1',
                    '-qmin', '1',
                    '-start_number', '0',
                    join(args.easyvolcap_root, scene, 'images', dirname) + '/%06d.jpg'
                ]
                subprocess.run(cmd, check=True)
            if basename(v).split('.')[0] != dirname:
                os.rename(v, join(os.path.dirname(v), dirname + '.mp4'))

        # https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/datasets/llff.py#L177
        raw = np.load(join(args.neural3dv_root, scene, args.camera_pose), allow_pickle=True)  # 21, 17
        poses = raw[:, :15].reshape(-1, 3, 5)  # N, 3, 5
        bounds = raw[:, -2:]  # N, 2
        # Step 1: rescale focal length according to training resolution
        H, W, F = poses[0, :, -1]  # original intrinsics, same for all images

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right down front"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], poses[..., :1], -poses[..., 2:3], poses[..., 3:4]], -1)  # (N_images, 3, 4) exclude H, W, focal
        cameras = dotdict()

        names = sorted(os.listdir(join(args.neural3dv_root, scene, 'images')))
        for i in range(len(poses)):
            key = names[i]

            c2w = poses[i]
            w2c = as_numpy_func(affine_inverse)(c2w)

            cameras[key] = dotdict()
            cameras[key].R = w2c[:3, :3]
            cameras[key].T = w2c[:3, 3:]
            cameras[key].K = np.zeros_like(cameras[key].R)
            cameras[key].K[0, 0] = F
            cameras[key].K[1, 1] = F
            cameras[key].K[0, 2] = W / 2
            cameras[key].K[1, 2] = H / 2
            cameras[key].K[2, 2] = 1.0
            # cameras[key].n = bounds[i, 0]  # camera has near and far
            # cameras[key].f = bounds[i, 1]  # camera has near and far

        write_camera(cameras, join(args.easyvolcap_root, scene))
        log(yellow(f'Converted cameras saved to {blue(join(args.easyvolcap_root, scene, "{intri.yml,extri.yml}"))}'))


def convert_colmap_ws_to_evc():
    """
    Assuming SfM is finished, convert the colmap point clouds to easyvolcap format.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='data/neural3dv/flame_salmon_1')
    parser.add_argument('--colmap_path', type=str, default='/usr/local/bin/colmap')
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    frames = [folder.name for folder in root_dir.glob('colmap_*') if folder.is_dir()]

    for frame in tqdm(frames):
        frame_dir = root_dir / frame
        sfm_dir = frame_dir / 'sparse' / '0'

        os.makedirs(root_dir / 'pcds', exist_ok=True)
        frame_id = frame.split('_')[1]
        cmd = [args.colmap_path, 'model_converter', '--input_path', str(sfm_dir), '--output_path', f'{root_dir}/pcds/{frame_id}.ply', '--output_type', 'PLY']
        # print(cmd)
        subprocess.call(cmd)


if __name__ == '__main__':
    main()
    # convert_colmap_ws_to_evc()
