"""
Load EasyVolcap format cameras, export colored plys
"""
import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.data_utils import export_camera
from easyvolcap.utils.math_utils import affine_inverse


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/volcano/skateboard')
    parser.add_argument('--intri_file', default='intri.yml')
    parser.add_argument('--extri_file', default='extri.yml')
    parser.add_argument('--axis_size', type=float, default=0.10)
    args = parser.parse_args()

    intri_path = join(args.data_root, args.intri_file)
    extri_path = join(args.data_root, args.extri_file)
    cams = read_camera(intri_path, extri_path)

    w2c = torch.stack([torch.concatenate([torch.tensor(c.R), torch.tensor(c.T)], dim=-1) for c in cams.values()])
    c2w = affine_inverse(w2c)
    ixt = torch.stack([torch.tensor(c.K) for c in cams.values()])

    cam_path = join(args.data_root, 'cameras.ply')
    export_camera(c2w, ixt, filename=cam_path, axis_size=args.axis_size)

    log(yellow(f'Camera visualization saved to {blue(cam_path)}'))


if __name__ == '__main__':
    main()
