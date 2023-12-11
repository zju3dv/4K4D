import os
import argparse

from os.path import join, split, abspath
from os import chdir, listdir, system, getcwd

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import *
# fmt: on


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/nas/home/xuzhen/datasets/xuzhen36/talk')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--mask_dir', type=str, default='mask')
    parser.add_argument('--schp_dir', type=str, default='schp')
    parser.add_argument('--chkpt_dir', type=str, default='/nas/home/xuzhen/weights/self-correction-human-parsing')
    args = parser.parse_args()
    args.data_root = abspath(args.data_root)
    tmp_human_dir = f'{args.data_root}/tmp_human'

    run_if_not_exists(f'python scripts/segmentation/merge_images_to_one_dir.py {args.data_root} {tmp_human_dir}/images/tmp_cam --one_human', f'{tmp_human_dir}/images/tmp_cam')
    cwd = getcwd()
    os.chdir(f'3rdparty/self-correction-human-parsing')

    run(f'python extract_multi.py {tmp_human_dir} --tmp {tmp_human_dir}/tmp --ckpt_dir {args.chkpt_dir}')
    for cam in listdir(f"{tmp_human_dir}/images"):
        run(f'mkdir -p {tmp_human_dir}/{args.schp_dir}')
        run(f'mkdir -p {tmp_human_dir}/{args.mask_dir}')

        run(f'mv {tmp_human_dir}/tmp/tmp_tmp_human_images_{cam}/mhp_fusion_parsing/global_parsing {tmp_human_dir}/{args.schp_dir}/{cam}')
        run(f'mv {tmp_human_dir}/tmp/tmp_tmp_human_images_{cam}/mhp_fusion_parsing/schp {tmp_human_dir}/{args.mask_dir}/{cam}')

    os.chdir(cwd)
    run(f'python scripts/segmentation/split_images_from_one_dir.py {args.data_root} {tmp_human_dir}/{args.mask_dir}/tmp_cam --mask {args.mask_dir} --one_human')
    run(f'python scripts/segmentation/split_images_from_one_dir.py {args.data_root} {tmp_human_dir}/{args.schp_dir}/tmp_cam --mask {args.schp_dir} --one_human')
    run(f'rm -rf {tmp_human_dir}')

    log(yellow(f'Mask saved to {blue(join(args.data_root, args.mask_dir))} and semantics saved to {blue(join(args.data_root, args.schp_dir))}'))


if __name__ == '__main__':
    main()
