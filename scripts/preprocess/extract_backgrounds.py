# This extracts the background images from the easyvolcap type datasets
# NOTE: This process could be extremely slow, as it needs to read the entire images and masks of the whole dataset

from os.path import join, exists
import imageio.v2 as imageio
import numpy as np
import argparse
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/renbody')
    parser.add_argument('--scenes', nargs='+', type=str, default=[])
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--masks_dir', type=str, default='masks')
    parser.add_argument('--bkgds_dir', type=str, default='bkgd')
    parser.add_argument('--uniform', type=bool, default=False)
    args = parser.parse_args()

    data_root = args.data_root
    scenes = args.scenes
    image_dir = args.image_dir
    masks_dir = args.masks_dir
    bkgds_dir = args.bkgds_dir
    uniform = args.uniform  # whether to use uniform background for all scenes

    if len(scenes) == 0:
        scenes = sorted(os.listdir(data_root))
        scenes = [scene for scene in scenes if exists(join(data_root, scene, image_dir))]        
    log(yellow(f'There are total {len(scenes)} scenes in {data_root}.'))

    # NOTE: Assume all scenes in the dataset share one set of cameras
    camera_names = sorted(os.listdir(join(data_root, scenes[0], image_dir)))
    # Need a list to store the background images only if use a uniform set of backgrounds for all scenes
    if uniform: bkgds = [None] * range(len(scenes))

    def process_background(i, camera):
        sh = imageio.imread(join(data_root, scenes[0], image_dir, camera, '000000.jpg')).shape
        bkgd = np.zeros(sh, dtype=np.uint8)

        def process_scene(scene):
            img_dir = join(data_root, scene, image_dir, camera)
            msk_dir = join(data_root, scene, masks_dir, camera)

            # Create the background directory
            if not uniform:
                os.makedirs(join(data_root, scene, bkgds_dir), exist_ok=True)
            else:
                if not exists(join(data_root, scene, bkgds_dir)):
                    os.system(f'ln -s {join(data_root, bkgds_dir)} {join(data_root, scene, bkgds_dir)}')

            for frame in os.listdir(img_dir):
                # Load image and mask
                img = imageio.imread(join(img_dir, frame))
                if not exists(join(msk_dir, frame)): msk = imageio.imread(join(msk_dir, frame.replace('.jpg', '.png')))
                else: msk = imageio.imread(join(msk_dir, frame))

                # Update background
                bkgd_mask = np.all(bkgd == 0, axis=-1)
                bkgd_mask = np.logical_and(bkgd_mask, msk == 0)
                bkgd[bkgd_mask] = img[bkgd_mask]

            # Update the background image
            if uniform: bkgds[i] = bkgd
            else: imageio.imwrite(join(data_root, scene, bkgds_dir, f'{camera}.jpg'), bkgd)

        parallel_execution(scenes, action=process_scene, sequential=True, print_progress=True)

    # Compute all the background images
    parallel_execution(list(range(len(camera_names))), camera_names, action=process_background, sequential=True, print_progress=True)

    # Write all the background images to the disk
    if uniform:
        os.makedirs(join(data_root, bkgds_dir))
        for cam, bkgd in zip(camera_names, bkgds):
            imageio.imwrite(join(data_root, bkgds_dir, f'{cam}.jpg'), bkgd)


if __name__ == '__main__':
    main()
