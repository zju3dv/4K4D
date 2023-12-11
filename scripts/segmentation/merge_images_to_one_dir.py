import os
import sys
import argparse
from os.path import join

# fmt: off
sys.path.append('.')
from easyvolcap.utils.console_utils import run
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)  # the root folder, whose images folder will be recursively liked to output
    parser.add_argument('output', type=str)  # the output folder
    parser.add_argument('--one_human', action='store_true')
    args = parser.parse_args()

    run(f'mkdir -p {args.output}')

    humans = os.listdir(args.path)
    if args.one_human:  # replace human
        humans = [os.path.split(args.path)[-1]]
        args.path = args.path.replace(humans[0], '')

    # path: should contain multiple humans, images in humans, cameras in images, and actual image in cameras
    img_paths = []
    new_img_paths = []
    for human in humans:
        human_dir = join(args.path, human)
        img_dir = join(human_dir, 'images')
        for cam in sorted(os.listdir(img_dir)):
            cam_dir = join(img_dir, cam)
            if cam_dir == args.output:
                continue
            for img in sorted(os.listdir(cam_dir)):
                img_path = join(cam_dir, img)
                new_img = f"{human}.{cam}.{img}"  # like F1_06_000000.02.000000.jpg
                new_img_path = join(args.output, new_img)
                # run(f'ln -s {img_path} {new_img_path}')
                img_paths.append(img_path)
                new_img_paths.append(new_img_path)
    parallel_execution(img_paths, new_img_paths, action=lambda x, y: run(f'ln -s {x} {y}', skip_failed=True))


if __name__ == '__main__':
    main()
