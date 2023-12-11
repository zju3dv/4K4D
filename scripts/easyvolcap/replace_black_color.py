import argparse
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, save_image

import os
import torch
import numpy as np
from glob import glob
from os.path import join, exists, basename, dirname


def process_one_image(pth: str,
                      out: str,
                      col: List = [0.267004, 0.004874, 0.329415],
                      ):
    img = load_image(pth)
    img = torch.from_numpy(img)  # H, W, 3/4
    col = torch.tensor(col)
    img[..., :3] = torch.where(img[..., :3] > 0, img[..., :3], col)
    img = img.numpy()
    save_image(out, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/novel_view/r4dv_0013_01_optcam/DEPTH')
    parser.add_argument('--output_dir', default='data/novel_view/r4dv_0013_01_optcam/DEPTH_filled')
    args = parser.parse_args()
    imgs = glob(join(args.input_dir, '*.png')) + glob(join(args.input_dir, '*.jpg'))
    outs = [join(args.output_dir, basename(img)) for img in imgs]
    parallel_execution(imgs, outs, action=process_one_image, print_progress=True)


if __name__ == '__main__':
    main()
