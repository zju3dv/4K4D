import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from glob import glob
from os.path import join
from tqdm import tqdm

from easymocap.bodymodel.smpl import SMPLModel
from easymocap.bodymodel.lbs import batch_rodrigues

from easyvolcap.utils.sh_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import load_mesh, load_dotdict, to_tensor
from easyvolcap.utils.net_utils import load_network
from easyvolcap.utils.gaussian_utils import GaussianModel
from easyvolcap.utils.easy_utils import load_bodymodel
from easyvolcap.utils.blend_utils import world_points_to_pose_points, pose_points_to_world_points
from easyvolcap.utils.sample_utils import sample_blend_K_closest_points

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.ops import knn_points


def get_transform(weights, A, eps=torch.finfo(torch.float32).eps, inverse=False):
    """
    weights: B, N, J
    A: B, J, D, D
    """
    T = torch.einsum('bpn,bnij->bpij', weights, A)
    dim = T.shape[-1]
    if inverse:
        T = (T + eps * torch.eye(dim, device=T.device, dtype=T.dtype)[None, None]).inverse()
    return T


def transform(xyz, T):
    xyz = F.pad(xyz, (0, 1), value=1.0)
    xyz = torch.einsum("bpij,bpj->bpi", T, xyz)[..., :3]
    return xyz


def load_pcd(path, sh_deg, smpl, prefix='sampler.pcds.0.', freeze=True, norm_with_smpl=True):
    pcd = GaussianModel(torch.rand(1, 3), None, 0.1, sh_deg)
    load_network(pcd, path, prefix=prefix)
    if norm_with_smpl:
        Rh = smpl['Rh']
        Th = smpl['Th']
        pcd._xyz.data = world_points_to_pose_points(pcd._xyz[None], Rh, Th)[0]
        pcd._xyz.grad = None
        R = quaternion_to_matrix(pcd._rotation)
        R = Rh[0].mT @ R
        pcd._rotation.data = matrix_to_quaternion(R)
        pcd._rotation.grad = None
    if freeze: 
        for params in pcd.parameters():
            params.requires_grad = False
    assert pcd.active_sh_degree.item() == sh_deg
    return pcd


def load_smpl(path):
    smpl = to_tensor(load_dotdict(path))
    smpl = dotdict({
        'shapes': smpl.shapes[:1],
        'poses': smpl.poses[:1],
        'Rh': batch_rodrigues(smpl.Rh[:1]),
        'Th': smpl.Th[:1],
    })
    return smpl


def compute_lbs(pcd: GaussianModel, smpl: dotdict, bodymodel: SMPLModel, K=4):
    xyz = pcd.get_xyz
    smpl_verts = bodymodel(shapes=smpl['shapes'], poses=smpl['poses'])
    weights, dists = sample_blend_K_closest_points(xyz[None], smpl_verts, bodymodel.weights[None], K=K)
    params = {'shapes': smpl['shapes'], 'poses': smpl['poses']}
    A, _ = bodymodel.transform(params)
    return weights, A.inverse()


def main():
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t', 'test'] + evv_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl_file', '-s',  type=str)
    parser.add_argument('--pcd_file', '-p', type=str)
    parser.add_argument('--out_dir', '-o', type=str, default='pre_gaussians')
    # some hyper parameters
    parser.add_argument('--sh_deg', type=int, default=0)
    parser.add_argument('--lbs_K', type=int, default=4)
    args = parser.parse_args(our_args)

    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg

    data_root = cfg.dataloader_cfg.dataset_cfg.data_root
    bodymodel_file = cfg.dataloader_cfg.dataset_cfg.bodymodel_file
    motion_file = cfg.dataloader_cfg.dataset_cfg.motion_file
    
    out_dir = join(data_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ref_smpl: dotdict = load_smpl(args.smpl_file)
    ref_pcd: GaussianModel = load_pcd(args.pcd_file, args.sh_deg, ref_smpl, 
                                      freeze=True, norm_with_smpl=True)
    bodymodel: SMPLModel = load_bodymodel(data_root, bodymodel_file)
    weights, inv_A = compute_lbs(ref_pcd, ref_smpl, bodymodel, args.lbs_K)

    motions = to_tensor(load_dotdict(join(data_root, motion_file)))
    if motions.Rh.ndim == 2:
        motions.Rh = batch_rodrigues(motions.Rh)
    As, _ = bodymodel.transform(motions)

    n_frames = motions.poses.shape[0]
    for i in tqdm(range(n_frames)):
        Rh = motions.Rh[i][None]
        Th = motions.Th[i][None]
        A = torch.einsum('bnij,bnjk->bnik', As[i][None], inv_A)
        T = get_transform(weights, A)

        xyz = ref_pcd.get_xyz[None]
        xyz = transform(xyz, T)
        xyz = pose_points_to_world_points(xyz, Rh, Th)[0]
        
        _rotation = ref_pcd._rotation
        R = quaternion_to_matrix(_rotation)
        R = torch.bmm(T[0, ..., :3, :3], R)
        R = torch.bmm(Rh.expand(R.shape[0], -1, -1), R)
        _rotation = matrix_to_quaternion(R)

        ret = {
            '_xyz': xyz.cpu().numpy(),
            '_rotation': _rotation.cpu().numpy()
        }
        np.savez(join(out_dir, f'{i:0>6d}.npz'), **ret)


if __name__ == "__main__":
    main()