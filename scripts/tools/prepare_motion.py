# this file will prepare a motion data for training or testing

import os
import json
import time
import torch
import pickle
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from os.path import join
from termcolor import colored
from multiprocessing import Process

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

# fmt: off
import sys

sys.path.append('.')
from easyvolcap.utils.console_utils import log, run, stacktrace
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import to_numpy, load_dotdict
from easyvolcap.utils.math_utils import affine_inverse
# fmt: on


def load_raw_amass(param_file: str):
    motion = load_dotdict(param_file)
    poses = motion.poses
    Th = motion.trans
    return poses.astype(np.float32), Th.astype(np.float32)


def load_amass_params(param_file: str, human: str, args):
    return load_aist_plusplus_params(param_file, human, args, loader=load_raw_amass)


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def load_raw_aist_plusplus(param_file: str):
    params = read_pickle(param_file)
    poses = params['smpl_poses']  # 640, 72
    Th = params['smpl_trans']  # 640, 3
    scaling = params['smpl_scaling']
    Th = Th / scaling

    return poses.astype(np.float32), Th.astype(np.float32)


def load_aist_plusplus_params(param_file: str, human: str, args, loader=load_raw_aist_plusplus):
    pivot = join(args.data_root, human, args.pivot_file)
    log(f'Loading pivot shape & trans & rotation from {colored(pivot, "blue")}')
    pivot_poses, pivot_Rh, pivot_Th, pivot_shapes = load_one_easymocap(pivot, args)  # pivot
    log(f'Loading motion shape & trans & rotation from {colored(param_file, "blue")}')
    poses, Th = loader(param_file)  # motion, N, 156; N, 3

    # Expand poses as needed (asit++ only provides SMPL)
    N = poses.shape[0]
    full = np.zeros((N, args.n_bones * 3), dtype=poses.dtype)
    full[:, :poses.shape[-1]] = poses
    full[:, :3] = 0  # set pose rotation to 0
    Rh = poses[:, :3]  # extract rotation, N, 3
    shapes = np.tile(pivot_shapes[None], (N, 1))

    # Rh extracted from poses are performed after Th
    # Thus we should first construct the full matrices from this
    # Originaly: RT @ ppts -> wpts
    # Assume they are affine transforms
    # We wang RT @ RT0.inverse() @ RT1 @ ppts -> wpts
    T = torch.from_numpy(Th[..., None])  # N, 3, 1
    R = axis_angle_to_matrix(torch.from_numpy(Rh))  # N, 3, 3
    P = R.new_zeros(N, 1, 4)
    P[..., -1] = 1.0
    A = torch.cat([R, T], dim=-1)  # N, 3, 4
    A = torch.cat([A, P], dim=-2)  # N, 4, 4

    # Construct the matrix for the first frame
    R0 = axis_angle_to_matrix(torch.from_numpy(Rh[:1]))  # in matrix
    P0 = R0.new_zeros(1, 1, 4)  # 1, 1, 4
    P0[..., -1] = 1.0
    T0 = torch.from_numpy(Th[:1, ..., None])  # 1, 3, 1
    A0 = torch.cat([R0, T0], dim=-1)
    A0 = torch.cat([A0, P0], dim=-2)

    # Construct the matrix for the pivot point
    R1 = axis_angle_to_matrix(torch.from_numpy(pivot_Rh[None]))  # in matrix
    P1 = R1.new_zeros(1, 1, 4)  # 1, 1, 4
    P1[..., -1] = 1.0
    T1 = torch.from_numpy(pivot_Th[None, ..., None])  # 1, 3, 1
    A1 = torch.cat([R1, T1], dim=-1)
    A1 = torch.cat([A1, P1], dim=-2)

    # This could be slow on cpu (a lot of matrix), especially the inverse part
    A = A1 @ affine_inverse(A0) @ A
    R = A[..., :3, :3]  # N, 3, 3
    T = A[..., :3, 3]  # N, 3

    Rh = to_numpy(matrix_to_axis_angle(R))
    Th = to_numpy(T)  # move to pivot center, after the full transformation

    return full, Rh, Th, shapes


def load_one_easymocap(input: str, args):
    ext = os.path.splitext(input)[-1]
    if ext == '.json':
        params = json.load(open(input))['annots'][0]
    elif ext == '.npy':
        params = np.load(input, allow_pickle=True).item()
    poses = np.array(params['poses'])[0]  # remove first dim
    shapes = np.array(params['shapes'])[0]  # remove first dim
    Rh = np.array(params['Rh'])[0]
    Th = np.array(params['Th'])[0]

    full_poses = np.zeros((args.n_bones * 3))
    full_poses[:poses.shape[-1]] = poses
    # full_poses[:3] = Rh

    # the params of neural body
    return full_poses.astype(np.float32), Rh.astype(np.float32), Th.astype(np.float32), shapes.astype(np.float32)


def load_easymocap_params(param_dir: str, human: str, args):
    param_files = sorted(os.listdir(param_dir), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    param_files = [join(param_dir, p) for p in param_files]
    poses, Rh, Th, shapes = zip(*parallel_execution(param_files, args, action=load_one_easymocap, print_progress=True))
    poses = np.stack(poses)
    Rh = np.stack(Rh)
    Th = np.stack(Th)
    shapes = np.stack(shapes)
    return poses, Rh, Th, shapes


def load_params_params(param_dir: str, human: str, args):
    return load_easymocap_params(param_dir, human, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/mobile_stage')
    parser.add_argument('--humans', type=str, nargs='+', default=['xuzhen'])
    parser.add_argument('--motion_in', type=str, default='easymocap/output-output-smpl-3d/smplfull')
    parser.add_argument('--motion_out', type=str, default='motion.npz')
    parser.add_argument('--motion_type', type=str, default='easymocap', choices=['easymocap', 'params', 'aist_plusplus', 'amass'])
    parser.add_argument('--n_bones', type=int, default=52, choices=[24, 52])

    # aist++ specific
    parser.add_argument('--pivot_file', type=str, default='easymocap/output-output-smpl-3d/smplfull/000000.json')
    args = parser.parse_args()

    for human_index in range(len(args.humans)):
        try:
            human = args.humans[human_index]
            if os.path.isfile(os.path.join(args.data_root, human)): continue  # skip files

            motion_in = os.path.join(args.data_root, human, args.motion_in)
            poses, Rh, Th, shapes = globals()[f'load_{args.motion_type}_params'](motion_in, human, args)
            np.savez_compressed(os.path.join(args.data_root, human, args.motion_out), poses=poses, Rh=Rh, Th=Th, shapes=shapes)

        except:
            stacktrace()
            continue  # do not stop for one error human


if __name__ == '__main__':
    main()
