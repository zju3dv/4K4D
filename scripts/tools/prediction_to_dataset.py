"""
This file will convert network prediction format to the actual dataset format

Network prediction format:
data/<result_dir>/<exp_name>/<save_tag>/<type>:
    frame<frame>_camera<camera><ext>
    frame<frame>_camera<camera><ext>

Dataset format:
data/<dataset>/<sequence>/<data>:
    <camera>/<frame><ext>
    <camera>/<frame><ext>
"""

from glob import glob
from easyvolcap.utils.console_utils import *


@catch_throw
def main():
    args = dotdict(
        result_dir='result',
        exp_name='gsmap_dtu_dpt_rgb_exr',
        save_tag='0008_01_obj',
        type='DEPTH',
        ext='.exr',
        dataset='renbody',
        sequence='0008_01',
        data='depths',
    )

    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    pred_dir = join('data', args.result_dir, args.exp_name, args.save_tag, args.type)
    data_dir = join('data', args.dataset, args.sequence, args.data)

    files = glob(join(pred_dir, f'*{args.ext}'))
    for f in tqdm(files):
        frame_str_idx = f.index('frame') + len('frame')
        camera_str_idx = f.index('camera') + len('camera')
        frame = int(f[frame_str_idx:frame_str_idx + 4])
        camera = int(f[camera_str_idx:camera_str_idx + 4])
        dst = join(data_dir, f'{camera:02d}', f'{frame:06d}' + args.ext)
        os.makedirs(dirname(dst), exist_ok=True)
        shutil.copy(f, dst)


if __name__ == '__main__':
    main()
