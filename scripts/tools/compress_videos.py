"""
Compress the video folder to something like libx265
"""

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import generate_video


@catch_throw
def main():
    args = dotdict(
        data_root='data/bullet/final',
        videos_dir='videos',
        output_dir='videos_compressed',
        vcodec='libx265',
        preset='veryslow',
        crf=23,
    )

    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    for video in sorted(os.listdir(join(args.data_root, args.videos_dir))):
        if not video.endswith('.mp4'): continue
        video_path = join(args.data_root, args.videos_dir, video)
        output_path = join(args.data_root, args.output_dir, video)
        os.makedirs(dirname(output_path), exist_ok=True)
        generate_video(video_path, output_path, fps=-1, verbose=True, crf=args.crf, vcodec=args.vcodec, preset=args.preset)


if __name__ == '__main__':
    main()
