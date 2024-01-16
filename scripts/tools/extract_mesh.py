"""
Load a easyvolcap model
Perform rendering on all images of a particular frame
Save the rendered rgb and depth value, along with maybe other attributes
Fuse the final rendered depth values as points into one
This function will try to invoke evc programmatically
"""
import torch
import argparse
from functools import partial
from os.path import join
import trimesh
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, point_padding
from easyvolcap.utils.mesh_utils import get_surface_sliding
from easyvolcap.utils.data_utils import to_cuda
from torch.utils.data import default_collate 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t', 'test'] + evv_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='data/geometry')
    parser.add_argument('--occ_thresh', type=float, default=0.0)
    parser.add_argument('--occ_scale', type=float, default=0.01)
    parser.add_argument('--dist_default', type=float, default=0.005)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--skip_align', action='store_true')
    parser.add_argument('--use_bounds', action='store_true')
    args = parser.parse_args(our_args)

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.engine import SAMPLERS
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    runner.load_network()
    runner.model.eval()
    extract(runner, args)  # pcd is a dotdict with all attributes that we want to retain


def extract(runner: "VolumetricVideoRunner", args: argparse.Namespace):
    result_dir = args.result_dir

    network = runner.model.network
    if hasattr(network, 'networks'):
        network = network.networks[-1]
    network = network.cuda()
    network.eval()
    dataset = runner.val_dataloader.dataset
    inds = runner.val_dataloader.batch_sampler.sampler.inds
    nv, nl = inds.shape[:2]
    prefix = 'frame'

    if dataset.closest_using_t:
        nv, nl = nl, nv
        prefix = 'view'
        inds = inds.transpose(0, 1)

    if not hasattr(network, 'sdf'):
        def func(x, d, batch): return (-getattr(network, 'occ')(x, d, args.dist_default, batch) + args.occ_thresh) * args.occ_scale
    else:
        func = partial(network.sdf, skip_parameterization=False, skip_deformation=False)

    for f in range(nl):
        batch = dataset[inds[0, f]]  # get the batch data for this view
        if args.use_bounds:
            bounds = batch.bounds.cpu().numpy()
        else:
            bounds = [[-2, -2, -2], [2, 2, 2]]
        batch = default_collate([batch])
        batch = to_cuda(batch)
        def sdf(x): return func(x, f * torch.ones_like(x[..., :1]), batch=batch)[..., 0]
        filename = join(result_dir, runner.exp_name, runner.visualizer.save_tag, 'MESH', f'{prefix}{f:04d}.ply')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        get_surface_sliding(sdf,
                            resolution=args.resolution,
                            bounding_box_min=bounds[0],
                            bounding_box_max=bounds[1],
                            level=args.occ_thresh,
                            output_path=filename,
                            simplify_mesh=True,
                            )
        if dataset.use_aligned_cameras and not args.skip_align:
            for name in [filename, filename.replace('.ply', '-simplify.ply')]:
                mesh = trimesh.load(name, process=False)
                vertices = torch.from_numpy(mesh.vertices).float()
                normals = torch.from_numpy(mesh.face_normals).float()
                mat = affine_padding(dataset.c2w_avg)
                vertices = (point_padding(vertices) @ mat.mT)[..., :3]
                normals = normals @ mat[..., :3, :3].mT
                mesh.vertices = vertices.cpu().numpy()
                mesh.face_normals = normals.cpu().numpy()
                mesh.export(name)

if __name__ == '__main__':
    main()
