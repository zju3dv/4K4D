"""
Find the images folder, rename it to images_flatten, and create a new images folder with the same structure as separated cameras
"""
# Rearrange images and camera parameters
import shutil
import argparse
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera, write_camera


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/zju/ipstage')
    parser.add_argument('--images_dirs', nargs='+', default=['images', 'masks'])
    parser.add_argument('--cameras_dir', default='cameras')
    parser.add_argument('--camera_dir', default='00')
    parser.add_argument('--image_base', default='000000')
    parser.add_argument('--extri_file', default='extri.yml')
    parser.add_argument('--intri_file', default='intri.yml')
    args = parser.parse_args()

    postfix = '_flatten'

    for image_dir in args.images_dirs:
        images_path = join(args.data_root, image_dir)

        if not exists(images_path):
            continue

        if len(os.listdir(images_path)) > 1:
            # raise ValueError(f'images_path {images_path} contains more than one cameras')
            continue

        try: os.rename(images_path, images_path + postfix)
        except: pass

        os.makedirs(images_path, exist_ok=True)
        for idx, cam in enumerate(sorted(os.listdir(images_path + postfix + f'/{args.camera_dir}'))):
            os.makedirs(join(images_path, f'{idx:06d}'), exist_ok=True)
            src = join(images_path + postfix, args.camera_dir, cam)
            tar = join(images_path, f'{idx:06d}', args.image_base + os.path.splitext(cam)[-1])
            try: os.remove(tar)
            except: pass
            os.rename(src, tar)

        shutil.rmtree(images_path + postfix, ignore_errors=True)

    cameras_path = join(args.data_root, args.cameras_dir)
    if exists(cameras_path):
        if len(os.listdir(cameras_path)) > 1:
            # raise ValueError(f'cameras_path {cameras_path} contains more than one cameras')
            pass

        try: os.rename(cameras_path, cameras_path + postfix)
        except: pass

        src = join(cameras_path + postfix, args.camera_dir, args.extri_file)
        tar = join(args.data_root, args.extri_file)
        try: os.remove(tar)
        except: pass
        os.rename(src, tar)

        src = join(cameras_path + postfix, args.camera_dir, args.intri_file)
        tar = join(args.data_root, args.intri_file)
        try: os.remove(tar)
        except: pass
        os.rename(src, tar)

        shutil.rmtree(cameras_path + postfix, ignore_errors=True)
    else:
        if not exists(join(args.data_root, args.intri_file)) and exists(join(args.data_root, args.extri_file)):
            raise ValueError(f'Unable to locate intri.yml and extri.yml or cameras in {args.data_root}')
        # else:
        #     cameras = read_camera(join(args.data_root, args.intri_file), join(args.data_root, args.extri_file))


if __name__ == '__main__':
    main()
