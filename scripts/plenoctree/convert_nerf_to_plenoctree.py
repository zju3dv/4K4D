# TODO: make this compatible with current pipeline

import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from svox import N3Tree
from os.path import join
from termcolor import cprint
from typing import Callable, Collection


# fmt: off
import sys
sys.path.append('.') # we assume you're calling the script from root of the repo

parser = argparse.ArgumentParser(description='''
NeRF like structure to PlenOctree conversion tool
''')
parser.add_argument('-c', '--cmd', default='python train_net.py -c configs/resd/jac.yaml')
parser.add_argument('-o', '--opt', default=[], nargs="*")
args = parser.parse_args()

options = args.opt
cmdargs = args.cmd
cmdargs = cmdargs.split()[1:]  # remove `python``
sys.argv = cmdargs + options

from easyvolcap.engine import cfg # using original network configuration
from easyvolcap.utils.sh_utils import ProjectFunctionNeRF, ProjectFunctionNeRFReuse
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.net_utils import load_network
from easyvolcap.utils.nerf_utils import alpha2raw
from easyvolcap.utils.data_utils import export_pts, export_mesh, to_tensor, to_cuda, add_batch
from easyvolcap.utils.sample_utils import sample_K_closest_points

# FIXME: FIX THESE IMPORTS
from easyvolcap.models.networks import make_network
from easyvolcap.models.networks import make_dataset
from easyvolcap.models.networks import Dataset
from easyvolcap.models.networks import Network
# fmt: on


@torch.no_grad()
def main():
    cprint('output: ', color='red', attrs=['bold'], end='')
    log(cfg.octree_path, 'yellow')

    log('making network')
    network: Network = make_network(cfg)
    network = network.cuda()
    network = network.eval()
    epoch = load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)  # load network into `network`

    log('getting dataset sample')
    dataset: Dataset = make_dataset(cfg, is_train=False)
    batch = to_tensor(dataset[cfg.octree.latent_index * dataset.num_cams])  # get a sample dataset, no batch dimension
    tbounds = batch.tbounds.cuda()  # 2, 3
    tverts = batch.tverts.cuda()  # 6890, 3
    center = (tbounds[1] + tbounds[0]) / 2  # 3,
    radius = (tbounds[1] - tbounds[0]) / 2
    # if cfg.octree.preserve_aspect_ratio:
    # radius[:] = radius.max()  # remove fancy scaling and leave samples where it's important

    log('creating octree structure')
    sh_dim = (cfg.octree.sh_deg + 1) ** 2
    data_dim = cfg.octree.channel * sh_dim + 1
    data_format = f'SH{sh_dim}'
    init_reserve = (2 ** cfg.octree.grid_depth) ** 2  # initially only reser on slab
    octree = N3Tree(N=cfg.octree.branch_factor,
                    data_dim=data_dim,
                    init_reserve=init_reserve,
                    depth_limit=cfg.octree.grid_depth,
                    radius=radius,
                    center=center,
                    data_format=data_format,
                    device='cuda')

    log("performing grid evaluation on sigma and dists")
    # create grid on cpu (too large to fit in VRAM)
    reso: torch.Tensor = 2 ** (cfg.octree.grid_depth + 1)
    offset = octree.offset.cpu()
    scale = octree.invradius.cpu()
    arr = (torch.arange(0, reso, dtype=torch.float32, device='cpu') + 0.5) / reso  # reso grid
    xx = (arr - offset[0]) / scale[0]
    yy = (arr - offset[1]) / scale[1]
    zz = (arr - offset[2]) / scale[2]
    # the grid might be too large to fit in the graphical memory
    grid = torch.stack(torch.meshgrid(xx, yy, zz)).reshape(3, -1).T  # 3, reso, reso, reso -> reso**3, 3
    log(f"initial grid shape: {grid.shape}")

    chunk_size = cfg.octree.chunk_size
    # filter by distance to smpl
    dists = []
    for i in tqdm(range(0, grid.shape[0], chunk_size)):
        grid_chunk = grid[i:i+chunk_size].cuda()
        dists_chunk = sample_K_closest_points(grid_chunk[None], tverts[None], K=1)[0][0, :, 0]  # add and remove fake batch dimension
        dists.append(dists_chunk)  # this value will be later discarded
    dists = torch.cat(dists, dim=0)  # reso ** 3,
    mask = dists < cfg.dist_th
    grid = grid[mask]  # first filter by smpl dists, might still be large

    # fileter by sigma
    alpha = []
    for i in tqdm(range(0, grid.shape[0], chunk_size)):
        grid_chunk = grid[i:i+chunk_size].cuda()
        alpha_chunk = network.occ.occ(grid_chunk[None])[0, :, 0]  # add and remove fake batch dimension
        alpha.append(alpha_chunk)  # this value will be later discarded
    alpha = torch.cat(alpha, dim=0)  # reso ** 3,

    mask = (alpha > cfg.octree.alpha_low_thresh) & (alpha < cfg.octree.alpha_high_thresh)
    grid = grid[mask]  # second filter by mask
    log(f'grid shape with valid alpha value: {grid.shape}')

    log('consturcting octree based on masked grid')
    grid = grid.cuda()  # move to GPU for octree queries
    refine_chunk_size = cfg.octree.refine_chunk_size
    for i in range(cfg.octree.grid_depth - 1):
        octree[grid].refine()  # will select and do refine
    if grid.shape[0] <= refine_chunk_size:
        octree[grid].refine()  # refine once if within computation budget
    else:
        for j in tqdm(range(0, grid.shape[0], refine_chunk_size)):
            octree[grid[j:j+refine_chunk_size]].refine()
    log(f'constructed octree: {octree}')
    assert octree.max_depth == cfg.octree.grid_depth, "error: actual grid depth and wanted grid depth mismatch"

    log("performing antialiasing & spherical harmonics evaluations")
    leaf_mask = octree.depths == octree.max_depth  # has batch dimension, n_node
    leaf_ind = torch.where(leaf_mask)[0]  # remove batch dimension, n_leaf,

    def cross_decoder(points: torch.Tensor, viewdirs: torch.Tensor):
        # points: [num_points, num_samples, 3]
        # viewdirs: [num_samples, num_rays, 3]

        # raw_rgb: [num_points, num_samples, num_rays, 3]
        # sigma: [num_points, num_samples, 1]

        n_points, n_samples, _ = points.shape
        n_samples, n_rays, _ = viewdirs.shape

        # occupancy network query, adding and removing batch dimension
        occ, feat, grad = network.occ(points)
        occ: torch.Tensor
        feat: torch.Tensor
        grad: torch.Tensor

        # rgb network query, using the selected latent index
        feat = feat[:, :, None].expand(-1, -1, n_rays, -1).view(n_points * n_samples, n_rays, -1)
        grad = grad[:, :, None].expand(-1, -1, n_rays, -1).view(n_points * n_samples, n_rays, -1)
        viewdirs = viewdirs[None, :, :].expand(n_points, -1, -1, -1).reshape(n_points * n_samples, n_rays, -1)
        rgb: torch.Tensor = network.rgb(viewdirs, grad, feat, cond)

        raw_rgb = rgb.logit().view(n_points, n_samples, n_rays, -1)
        sigma = alpha2raw(occ).view(n_points, n_samples, -1)

        return raw_rgb, sigma  # yes, rgb first

    chunk_size = cfg.octree.render_chunk_size // (cfg.octree.samples_per_cell * cfg.octree.projection_samples)
    cond = network.get_cond(grid, add_batch(to_cuda(batch)))  # grid is only a dummy input, not used

    for i in tqdm(range(0, leaf_ind.shape[0], chunk_size)):
        chunk_inds = leaf_ind[i:i+chunk_size]
        points = octree[chunk_inds].sample(cfg.octree.samples_per_cell)  # (n_cells, n_samples, 3)
        coeffs, alpha = project_nerf_to_sh(cross_decoder, cfg.octree.sh_deg, points)
        coeffs, alpha = coeffs.view(coeffs.shape[0], -1), alpha.view(alpha.shape[0], -1)

        coeffs_sigma = torch.cat([coeffs, alpha], dim=-1)
        coeffs_sigma = coeffs_sigma.view(-1, octree.data_dim)
        octree[chunk_inds] = coeffs_sigma

    cprint(f'max sigma: {octree[:, -1:].max()}\nmin sigma: {octree[:, -1:].min()}', color='magenta', attrs=['bold'])

    cprint('saving: ', color='cyan', attrs=['bold'], end='')
    log(cfg.octree_path)
    os.system(f'mkdir -p {os.path.dirname(cfg.octree_path)}')
    octree.save(cfg.octree_path)  # will: shrink + compress


def project_nerf_to_sh(decoder: Callable[[torch.Tensor, torch.Tensor], Collection[torch.Tensor]], sh_deg: int, points: torch.Tensor, projection_samples: int = cfg.octree.projection_samples):
    """
    Args:
        points: [n_points, n_samples, 3]
    Returns:
        coeffs for rgb. [n_points, C * (sh_deg + 1)**2]
    """

    def spherical_function(viewdirs: torch.Tensor):
        # points: [num_points, num_samples, 3]
        # viewdirs: [num_samples, num_rays, 3]

        # raw_rgb: [num_points, num_samples, num_rays, 3]
        # sigma: [num_points, num_samples, 1]
        raw_rgb, sigma = decoder(points, viewdirs)
        return raw_rgb, sigma

    coeffs, sigma = ProjectFunctionNeRFReuse(
        order=sh_deg,
        spherical_function=spherical_function,
        n_points=points.shape[0],
        n_samples=points.shape[1],
        n_rays=projection_samples,
        device=points.device)

    coeffs, sigma = coeffs.view(coeffs.shape[0], -1), sigma.view(sigma.shape[0], -1)

    return coeffs, sigma  # num_points, C, sh_dim; num_points, 1


if __name__ == '__main__':
    main()
