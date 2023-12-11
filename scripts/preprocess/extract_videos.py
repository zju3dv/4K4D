import os
import argparse
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/volcano')
    parser.add_argument('--videos_dir', default='videos')
    parser.add_argument('--images_dir', default='images')
    parser.add_argument('--cmd', default='/usr/bin/ffmpeg -i {video_path} -q:v 1 -qmin 1 -start_number 0 {output_path}/%06d.jpg')
    args = parser.parse_args()

    videos_dir = join(args.data_root, args.videos_dir)
    images_dir = join(args.data_root, args.images_dir)
    for vid in os.listdir(videos_dir):
        out = join(images_dir, os.path.splitext(vid)[0])
        vid = join(videos_dir, vid)
        os.makedirs(out, exist_ok=True)
        cmd = args.cmd.format(video_path=vid, output_path=out)
        run(cmd)


if __name__ == '__main__':
    main()
