"""
Given sequence dataset, select some of the frames to form a new sequence dataset.
# TODO: Support view sampling
"""

from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/enerf_outdoor/actor1_4')
    parser.add_argument('--out_root', default='data/enerf_outdoor/actor1_4_subseq')
    parser.add_argument('--frame_sample', default=[0, 150, 5], nargs='*', type=int)
    parser.add_argument('--dirs', default=['images', 'bgmtv2', 'surfs'], nargs='*')
    parser.add_argument('--copy', default=['optimized', 'bkgd'], nargs='*')
    args = parser.parse_args()

    b, e, s = args.frame_sample

    for d in args.dirs:
        os.makedirs(join(args.out_root, d), exist_ok=True)
        files = os.listdir(join(args.data_root, d))
        files = sorted(files)
        if all([os.path.isdir(join(args.data_root, d, f)) for f in files]):
            for cam in files:
                os.makedirs(join(args.out_root, d, cam), exist_ok=True)
                cam_files = os.listdir(join(args.data_root, d, cam))
                cam_files = sorted(cam_files)
                cam_files = cam_files[b:e:s]
                for i, f in enumerate(tqdm(cam_files, d)):
                    ext = splitext(f)[-1]
                    shutil.copy(join(args.data_root, d, cam, f), join(args.out_root, d, cam, f'{i:06d}{ext}'))
        else:
            files = files[b:e:s]
            for i, f in enumerate(tqdm(files, d)):
                ext = splitext(f)[-1]
                shutil.copy(join(args.data_root, d, f), join(args.out_root, d, f'{i:06d}{ext}'))

    for c in args.copy:
        if isdir(join(args.data_root, c)):
            shutil.copytree(join(args.data_root, c), join(args.out_root, c))
        else:
            shutil.copy(join(args.data_root, c), join(args.out_root, c))


if __name__ == '__main__':
    main()
