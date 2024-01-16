import os
import sys
import json
import torch
import argparse
import numpy as np
import ujson as json

from os.path import join
from torch.optim import Adam
from functools import partial
from bvh_distance_queries import BVH
from pytorch3d.structures import Meshes

# setup SMPL model based on user preference and easymocap initialization, like whether to use hand / pose PCA
from easymocap.config.baseconfig import load_object, Config
from easymocap.bodymodel.smplx import SMPLHModelEmbedding

from typing import Union, List

# fmt: off
# when invoking the script with python tools/fitSMPL.py, we need to extend PATH to actually be able to import these project wise utility functions
sys.path.append('.')
from easyvolcap.utils.console_utils import magenta, blue
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.chunk_utils import multi_gather, multi_gather_tris
from easyvolcap.utils.data_utils import export_mesh, export_dotdict, load_mesh
from easyvolcap.runners.schedulers import ExponentialLR
from scripts.torchxpbd.unproject_semantic_parsing import semantic_dim, semantic_list
# fmt: on

# utility functions for optimization


def select_params(params: dotdict, indices: Union[slice, int] = slice(None), fixed=[], only=[]):
    # select ith parameter or use a slice to select all of them
    # select on the batch dimension
    # if given an integer, will augment the parameters selected with an empty dim
    # fixed parameters are those that shouldn't be selected by expanded

    # always retain a batch dimension
    indices = [indices] if isinstance(indices, int) else indices

    # select regular parameters
    regular_params = {k: v[indices]for k, v in params.items() if k not in fixed}

    # unable to get the actual length from indices, might just be a plain old slice with no length unless acted upon
    B = next(iter(regular_params.values())).shape[0]

    # select fixed parameter
    fixed_params = {k: v[0][None].expand(B, *v.shape[1:]) for k, v in params.items() if k in fixed}

    # merge selected parameters
    params = dotdict({**regular_params, **fixed_params, })

    if not only:  # if we passed in an empty list of only, we would retain everyone
        only = params.keys()

    # update parametesr to retain only "only" parameters
    empty_params = {k: torch.zeros_like(v) for k, v in params.items() if k not in only}
    only_params = {k: v for k, v in params.items() if k in only}
    params = dotdict({**empty_params, **only_params, })

    return params


def register_smplh(params: dotdict[str, torch.Tensor], bodymodel: SMPLHModelEmbedding, tar_meshes: Meshes, batch: dotdict[str, torch.Tensor], **kwargs):
    """
    Register smplh model to a target mesh, accelerated with CUDA BVH, using Non-Rigid ICP Algorithm:
        poses:      n_bones * 3, initial pose estimation
        shapes:     n_shapes, initial shape estimation
        Rh:         3, initial global rotation estimation
        Th:         3, initial global translation estimation
        bodymodel:  SMPL model to compute vertices from poses, shapes, Rh, Th
        verts:      n_verts, 3, vertices of target mesh
        faces:      n_faces, 3, faces of target mesh
        kwargs:     addtional configuration, can overwrite the default values
    """

    # default configuration
    config = dotdict()
    config.lr = 1e-2
    config.gamma = 1e-1
    config.iter = 4000
    config.ep_iter = 50
    config.dist_th = 0.10  # 5cm
    config.angle_th = np.cos(np.deg2rad(45.0))
    config.inside_th = 0.0
    config.outside_lw = 2.0
    config.inside_lw = 0.25
    config.onbody_lw = 4.0
    config.onbody_type = [
        'hair',
        'face',
        'sock',
        'jumpsuit',
        'left_leg',
        'right_leg',
        'left_arm',
        'right_arm',
    ]
    config.fix_shape = True
    config.update(**kwargs)
    B = len(params.poses)  # batch size: number of shapes

    # prepare body semantic registration parameters
    body_indices = list(map(lambda x: semantic_list.index(x), config.onbody_type))
    body_indices = torch.tensor(body_indices, device=bodymodel.device, dtype=torch.long)

    # utility functions for registration
    select_full_params = partial(select_params, fixed=['shapes'] if config.fix_shape else [])
    select_shape_params = partial(select_full_params, only=['shapes'])

    # identity
    smplh_faces = torch.tensor(bodymodel.faces, dtype=torch.long, device=bodymodel.device)
    src_meshes = Meshes(torch.zeros(B, *bodymodel.shapedirs.shape[:2], dtype=torch.float, device=bodymodel.device), faces=smplh_faces[None].expand(B, *smplh_faces.shape))

    # optimizable parameters
    params = {k: v.requires_grad_(True) for k, v in params.items()}
    params = dotdict(params)

    # optimizer
    optim = Adam(params.values(), lr=config.lr)
    sched = ExponentialLR(optim, decay_iter=config.iter, gamma=config.gamma)

    # target mesh data preparations
    bvh = BVH()
    verts_padded = tar_meshes.verts_padded()  # B, VT, 3
    faces_padded = tar_meshes.faces_padded()  # B, F, 3
    normals_padded = tar_meshes.faces_normals_padded()  # B, VT, 3
    # tris_padded = multi_gather(verts_padded, faces_padded.view(*faces_padded.shape[:-2], -1)).view(*faces_padded.shape, -1) # B, F, 3, 3
    # tris_padded = verts_padded[torch.arange(B)[..., None, None], faces_padded] # B, F, 3, 3
    tris_padded = multi_gather_tris(verts_padded, faces_padded)  # B, F, 3, 3

    # logging
    max_key_len = max([len(str(key)) for key in config.keys()])
    max_val_len = max([len(str(val)) for val in config.values()])
    info = "\n".join([f'{k+":":<{max_key_len+1}} {magenta(v):<{max_val_len}}' for k, v in config.items()])
    print(f"beginning registration with config:\n{(info)}")

    for iter in range(config.iter):

        # forward pass throught the SMPLH model
        expanded_param = select_full_params(params)
        smplhs: torch.Tensor = bodymodel(**expanded_param)  # smplh wants a batch dimension, we don't want that

        dists_sq, points, face_ids, barys = bvh(tris_padded, smplhs)  # forward distance, find closest point on tris_padded of every smplhs vertices

        src: Meshes = src_meshes.offset_verts(smplhs.view(-1, smplhs.shape[-1]))
        src_normals = src.verts_normals_padded()  # B, VS, 3

        # use distance and angle fileter to get a good estimation
        filter0 = dists_sq < config.dist_th ** 2  # distance to the closest surface should be within range
        tar_normals = multi_gather(normals_padded, face_ids)
        direction = (tar_normals * src_normals).sum(dim=-1)  # dot product of tar_src surface normal
        filter1 = direction > config.angle_th  # angle between tar_src surface normal should be within range
        filter = filter0 & filter1

        delta = smplhs - points
        direction = (delta * src_normals).sum(dim=-1)  # note: not unit vector anymore

        # bit magic for body regions
        tar_semantics = multi_gather(batch.faces_semantics, face_ids)  # B, VS
        tar_semantics_bits = 1 << tar_semantics  # remember the semantic classes with bits
        body_indices_bits = (1 << body_indices).sum()  # no repeatition

        onbody = ((body_indices_bits & tar_semantics_bits) != 0) & filter
        outside = (direction > config.inside_th) & filter & ~onbody
        inside = (direction <= config.inside_th) & filter & ~onbody

        delta = delta ** 2  # L2 loss
        # MARK: CPU & GPU SYNC: PROFILING
        loss: torch.Tensor = \
            config.outside_lw * delta[outside].mean() + \
            config.inside_lw * delta[inside].mean() + \
            config.onbody_lw * delta[onbody].mean()

        # optimization step
        optim.zero_grad()
        loss.backward()
        optim.step()

        # logging & lr scheduler
        if not iter % config.ep_iter:
            sched.step()

            print(f'iter:\t{blue(f"{iter}/{config.iter}")}')
            print(f'loss:\t{magenta(f"{loss.item()}")}')

    params = {k: v.detach().requires_grad_(False) for k, v in params.items()}

    return params


"""
python scripts/torchxpbd/register_smplh_semantically.py -d data/xuzhen36/talk/registration/deformation -t data/xuzhen36/talk/uvmap.obj -i data/xuzhen36/talk/circular_ply/xuzhen_near2_01/output-static/smpl/000000.json -c data/xuzhen36/talk/circular_ply/xuzhen_near2_01/output-static/cfg_model.yml
"""


def load_init_params(init_params: List[str], device='cuda', store_init_smpl: bool = False, output_dirs: str = None, bodymodel: SMPLHModelEmbedding = None):
    # load init parameters from easymocap images space prediction, and the target mesh respectively
    keys = ["poses", "shapes", "Rh", "Th", 'handl', 'handr']
    all_params = {k: [] for k in keys}

    # load initial parameters: all_params
    for i in range(len(init_params)):

        # load init parameters from easymocap
        init_params = json.load(open(init_params[i]))  # not the best practice I admit
        init_params = {k: torch.tensor(np.array(init_params['annots'][0][k]), dtype=torch.float, device=device) for k in keys}  # assuming a batch dimension and retain it

        if store_init_smpl:
            smplh = bodymodel(**init_params)
            os.system(f"mkdir -p {output_dirs[i]}")
            export_mesh(smplh, bodymodel.faces, filename=join(output_dirs[i], 'init_smplh.ply'))

        for k, v in all_params.items():
            v.append(init_params[k])

    # create initial parameters to be optimized
    params = {k: torch.cat(v) for k, v in all_params.items()}
    params = dotdict(params)

    return params


def load_tar_meshes(target_meshes: List[str], device='cuda'):
    # load target meshes: tar_meshes
    verts, faces = [], []
    for tar in target_meshes:
        v, f = load_mesh(tar)
        verts.append(v)
        faces.append(f)
    tar_meshes = Meshes(verts, faces)
    return tar_meshes


def load_semantics(target_meshes: List[str], device='cuda'):
    # since the semantic information should have been loaded from the npz file directly
    # but due to the separated structure of the npz files, there won't be too much performance hit
    verts_semantics = []
    faces_semantics = []
    for tar in target_meshes:
        tpose = np.load(tar)
        vs = torch.tensor(tpose['verts_semantics'], device=device)
        fs = torch.tensor(tpose['faces_semantics'], device=device)
        faces_semantics.append(fs)
        verts_semantics.append(vs)
    return torch.stack(faces_semantics), torch.stack(verts_semantics)  # B, V,


def main():
    """
    This tool is for fitting SMPL-H to multiple poses of a person's mesh, simultaneously
    A SMPL-H model is controlled by `shapes`, `poses`, `Rh`, `Th` and optionally `handl` and `handr` parameters
    - We provide interface for using hand PCA components, defined by the MANO model, in which case you'll be optimizing the PCA weights of stored in handl and handr instead of the raw poses
    - We also provide interface for using body PCA, defined in `human_pose_prior`, but this might lead to unwanted pose constraints so it's disabled by default
    - We initialize the optimization with `easymocap` results, this effectively removes the need for hand-labeling landmarks
    - We optimize the parameters mentioned above using Non-Rigid ICP algorithm, together with some bias towards the hand region

    Also, as an interface file, this file also contains entry point to:
    - Blend weight extraction (using blender): `tools/genBW.py`
    - T-pose human mesh & data generation: `tools/genTPOSE.py`
    - Node graph (deformation graph) generation: `tools/genGRAPH.py`
    These will be called if you specify --postprocess when calling `tools/fitSMPL.py`
    """

    # scripts interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dirs', nargs='*', default=[
        'data/xuzhen36/talk/registration/deformation',
    ])
    parser.add_argument('-t', '--target_meshes', nargs='*', default=[
        'data/xuzhen36/talk/registration/deformation/semantic_tpose.npz',
    ])
    parser.add_argument('-i', '--init_params', nargs='*', default=[
        'data/xuzhen36/talk/circular_ply/xuzhen_near2_01/output-static/smpl/000000.json',
    ])

    # custom smpl options, if using easymocap-public, ignore these
    parser.add_argument('-c', '--cfg_model',  type=str, default='data/xuzhen36/talk/circular_ply/xuzhen_near2_01/output-static/cfg_model.yml')

    # device configuration
    parser.add_argument('-d', '--device', type=str, default='cuda')

    # other options
    parser.add_argument('opts', default=[], nargs=argparse.REMAINDER)

    # prepare args and config, anything unrelated to io will be stored in config instead of args
    args = parser.parse_args()
    opts = {args.opts[i]: args.opts[i+1] for i in range(0, len(args.opts), 2)}
    config = dotdict()
    config.update(opts)

    # load smpl model (maybe copy to gpu)
    cfg_model = Config.load(args.cfg_model)
    bodymodel: SMPLHModelEmbedding = load_object(cfg_model.module, cfg_model.args)
    bodymodel.to(args.device, non_blocking=True)

    # load init parameters
    init_params = load_init_params(args.init_params, args.device, store_init_smpl=True, output_dirs=args.output_dirs, bodymodel=bodymodel)

    # load target meshes: pytorch3d meshes
    tar_meshes = load_tar_meshes(args.target_meshes, args.device)

    # load semantic information
    faces_semantics, verts_semantics = load_semantics(args.target_meshes, args.device)

    batch = dotdict()
    batch.faces_semantics = faces_semantics
    batch.verts_semantics = verts_semantics

    # Non-Rigid ICP registration: optimize SMPL-H parameters to let it fit onto the target mesh
    params = register_smplh(init_params, bodymodel, tar_meshes, batch, **config)

    # store everything & maybe calling the postprocess steps
    smplhs: torch.Tensor = bodymodel(**params)
    for i in range(len(args.output_dirs)):
        # select corresponding mesh
        smplh = smplhs[i]

        # save optimzed smpl mesh
        export_mesh(smplh, bodymodel.faces, filename=join(args.output_dirs[i], 'optim_smplh.ply'))

        # save optimized smpl parameters
        optim_params = {k: v[i][None] for k, v in params.items()}
        optim_params['smpl_poses'] = optim_params['poses']
        optim_params['poses'] = bodymodel.export_full_poses(**optim_params)
        export_dotdict(optim_params, filename=join(args.output_dirs[i], 'optim_params.npz'))


if __name__ == '__main__':
    main()
