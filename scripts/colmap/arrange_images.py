"""
Collect images from images/00/000000.jpg to colmap/images/00.jpg
"""

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict()
    args.data_root = 'data/neural3dv/sear_steak'
    args.images_dir = 'images'
    args.image_idx = 0
    args.colmap_dir = 'colmap'
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    images_dir = join(args.data_root, args.images_dir)
    colmap_images_dir = join(args.data_root, args.colmap_dir, args.images_dir)
    os.makedirs(colmap_images_dir, exist_ok=True)

    image = sorted(os.listdir(join(args.data_root, args.images_dir, os.listdir(images_dir)[0])))[args.image_idx]
    ext = image.split('.')[-1]

    for cam in os.listdir(images_dir):
        src = join(images_dir, cam, image)
        tar = join(colmap_images_dir, f'{cam}.{ext}')

        if exists(tar): os.remove(tar)
        os.symlink(relpath(src, dirname(tar)), tar)

    log(yellow(f'Arranged images: {blue(colmap_images_dir)}'))


if __name__ == '__main__':
    main()
