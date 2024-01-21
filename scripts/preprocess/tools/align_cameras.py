# This script is used to perform camera alignment for a given `easyvolcap` format dataset.
# Namely, it does the same things as in `VolumetricVideoDataset.align_points()`, this script
# is just a standalone version of that function for you to export the aligned cameras.

import torch
import argparse
import numpy as np
from os.path import join

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import as_torch_func
from easyvolcap.utils.cam_utils import average_c2ws, average_w2cs
from easyvolcap.utils.easy_utils import read_camera, write_camera, to_easymocap
from easyvolcap.utils.math_utils import affine_inverse, affine_padding
from easyvolcap.utils.bound_utils import monotonic_near_far


def load_align_cameras(data_root: str, intri_file: str, extri_file: str, cameras_dir: str = 'cameras',
                       n_frames_total: int = 1, near: float = 0.2, far: float = 100.0,
                       avg_using_all: bool = False, avg_max_count: int = 100):

    # Multiview dataset loading, need to expand, will have redundant information
    if exists(join(data_root, intri_file)) and exists(join(data_root, extri_file)):
        cameras = read_camera(join(data_root, intri_file), join(data_root, extri_file))
        camera_names = np.asarray(sorted(list(cameras.keys())))  # NOTE: sorting camera names
        cameras = dotdict({k: [cameras[k] for i in range(n_frames_total)] for k in camera_names})
    # Monocular dataset loading, each camera has a separate folder
    elif exists(join(data_root, cameras_dir)):
        camera_names = np.asarray(sorted(os.listdir(join(data_root, cameras_dir))))  # NOTE: sorting here is very important!
        cameras = dotdict({
            k: [v[1] for v in sorted(
                read_camera(join(data_root, cameras_dir, k, intri_file),
                            join(data_root, cameras_dir, k, extri_file)).items()
            )] for k in camera_names
        })
    # Whatever else, for now, raise error
    else: raise NotImplementedError(f'Could not find [intri.yml, extri.yml] or [cameras] folder in {data_root}, check your dataset configuration')

    # cameras: a mapping from camera names to a list of camera objects, (every element in list is an actual camera for that particular view and frame)
    Hs = torch.as_tensor([[cam.H for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F)
    Ws = torch.as_tensor([[cam.W for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F)
    Ks = torch.as_tensor([[cam.K for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F, 3, 3)
    Rs = torch.as_tensor([[cam.R for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F, 3, 3)
    Ts = torch.as_tensor([[cam.T for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F, 3, 1)
    Ds = torch.as_tensor([[cam.D for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F, 1, 5)
    ts = torch.as_tensor([[cam.t for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F) # UNUSED: time index from camera, not used for now
    ns = torch.as_tensor([[cam.n for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F)
    fs = torch.as_tensor([[cam.f for cam in cameras[k]] for k in camera_names], dtype=torch.float)  # (V, F)
    w2cs = torch.cat([Rs, Ts], dim=-1)  # (V, F, 3, 4)
    c2ws = affine_inverse(w2cs)  # (V, F, 3, 4)
    ns, fs = monotonic_near_far(ns, fs, torch.as_tensor(near, dtype=torch.float), torch.as_tensor(far, dtype=torch.float))

    # Move cameras to the center of the frame (!: intrusive)
    c2ws, w2cs, Rs, Ts, c2w_avg = align_points(c2ws, avg_using_all, avg_max_count)

    # Return the aligned cameras
    return Ks, Hs, Ws, Rs, Ts, ts, ns, fs, Ds


def align_points(c2ws: torch.Tensor, avg_using_all: bool = False, avg_max_count: int = 100):
    sh = c2ws.shape  # (V, F, 3, 4)
    c2ws = c2ws.view((-1,) + sh[-2:])  # (V*F, 3, 4)

    if avg_using_all:
        stride = max(len(c2ws) // avg_max_count, 1)
        inds = torch.arange(len(c2ws))[::stride][:avg_max_count]
        c2w_avg = as_torch_func(average_c2ws)(c2ws[inds])  # (V*F, 3, 4), # !: HEAVY
    else:
        c2w_avg = as_torch_func(average_c2ws)(c2ws.view(sh)[:, 0])  # (V, 3, 4)
    c2w_avg = c2w_avg

    c2ws = (affine_inverse(affine_padding(c2w_avg))[None] @ affine_padding(c2ws))[..., :3, :]  # (1, 4, 4) @ (V*F, 4, 4) -> (V*F, 3, 4)
    w2cs = affine_inverse(c2ws)  # (V*F, 3, 4)
    c2ws = c2ws.view(sh)
    w2cs = w2cs.view(sh)
    Rs = w2cs[..., :-1]
    Ts = w2cs[..., -1:]

    return c2ws, w2cs, Rs, Ts, c2w_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/webcam/simple/light/calib_gather_230928/colmap/align/static/images')
    parser.add_argument('--intri_file', type=str, default='intri.yml')
    parser.add_argument('--extri_file', type=str, default='extri.yml')
    parser.add_argument('--cameras_dir', type=str, default='cameras')

    parser.add_argument('--n_frames_total', type=int, default=1)
    parser.add_argument('--near', type=float, default=0.25)
    parser.add_argument('--far', type=float, default=2.00)
    parser.add_argument('--avg_using_all', action='store_true')
    parser.add_argument('--avg_max_count', type=int, default=100)

    parser.add_argument('--cam_digit', type=int, default=1)
    parser.add_argument('--save_root', type=str, default='data/webcam/simple/light/calib_gather_230928/aligned')
    args = parser.parse_args()

    # Load and align cameras
    Ks, Hs, Ws, Rs, Ts, ts, ns, fs, Ds = load_align_cameras(
        args.data_root, args.intri_file, args.extri_file, args.cameras_dir,
        args.n_frames_total, args.near, args.far, args.avg_using_all, args.avg_max_count
    )

    # Convert loaded and aligned cameras to `EasyMocap` format
    # TODO: support for monocular cameras
    cameras = to_easymocap(Ks, Hs, Ws, Rs, Ts, ts, ns, fs, Ds, cam_digit=args.cam_digit)

    # Write cameras to disk
    write_camera(cameras, args.save_root)


if __name__ == '__main__':
    main()
