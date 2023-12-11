import os
import argparse
from os.path import join
from easyvolcap.utils.console_utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/mobile_stage/xuzhen')
    parser.add_argument('--output_dir', default='samples')
    parser.add_argument('--image_dir', default='images')
    parser.add_argument('--image_file', default='000000.jpg')
    args = parser.parse_args()

    cams = os.listdir(join(args.input, args.image_dir))
    os.makedirs(join(args.input, args.output_dir), exist_ok=True)
    log(f'Gathering images from {blue(join(args.input, args.image_dir))} to {blue(join(args.input, args.output_dir))}')
    for cam in tqdm(cams):
        os.system('cp {} {}'.format(
            join(args.input, args.image_dir, cam, args.image_file),
            join(args.input, args.output_dir, cam + '.jpg')
        ))

if __name__ == '__main__':
    main()