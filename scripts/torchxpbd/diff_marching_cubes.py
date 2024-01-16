import torch
from tqdm import tqdm
from torch.optim import Adam
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures import Meshes

# FIXME: FIX THESE IMPORTS
from lib.networks.deform.base_network import SignedDistanceNetwork
from lib.networks.deform.base_network import ResidualDeformation
from lib.networks.deform.bijective_network import BijectiveDeformation

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.bound_utils import get_bounds
from easyvolcap.utils.net_utils import take_gradient
from easyvolcap.utils.loss_utils import eikonal, l2_reg
from easyvolcap.utils.data_utils import export_mesh, load_mesh
from easyvolcap.utils.mesh_utils import laplacian_smoothing, differentiable_marching_cubes


log('preparing network')
sdf = SignedDistanceNetwork().to('cuda', non_blocking=True)

log('loading network')
input_file = 'data/trained_model/deform/monosem_ddp8/latest.pth'
pretrained = torch.load(input_file, map_location='cuda')
pretrained = pretrained['net']
prefix = 'signed_distance_network'
model = {}
for key, value in pretrained.items():
    if key.startswith(prefix):
        model[key[len(prefix) + 1:]] = value
sdf.load_state_dict(model)

log('generating data')
input_mesh = 'data/animation/deform/monosem_ddp8/can_mesh.npz'
verts, faces = load_mesh(input_mesh, device='cuda')
bounds = get_bounds(verts[None])[0]
voxel_size = [0.005, 0.005, 0.005]
x = torch.arange(bounds[0, 0], bounds[1, 0] + voxel_size[0], voxel_size[0], device='cuda')
y = torch.arange(bounds[0, 1], bounds[1, 1] + voxel_size[1], voxel_size[1], device='cuda')
z = torch.arange(bounds[0, 2], bounds[1, 2] + voxel_size[2], voxel_size[2], device='cuda')
pts = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)

lr = 1e-3
iter_cnt = 20
optim = Adam(sdf.parameters(), lr=lr)
for i in tqdm(range(iter_cnt)):
    verts, faces = differentiable_marching_cubes(pts.requires_grad_(), sdf.sdf)  # !: hacky way to mark requiring grads
    vsdf = sdf.sdf(verts)
    norm = take_gradient(vsdf, verts)
    data_loss_verts = (verts.norm(dim=-1) - 0.5).abs()
    data_loss = data_loss_verts.mean()
    grad_loss = eikonal(norm)
    loss = data_loss + grad_loss * 0.1
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    vsdf_mean = vsdf.abs().mean().item()

    log(f'iter: {i}')
    log(f'data_loss: {data_loss.item()}')
    log(f'grad_loss: {grad_loss.item()}')
    log(f'vsdf_mean: {vsdf_mean}', 'red' if vsdf_mean > 1e-4 else 'green')

    export_mesh(verts, faces, colors=colormap(data_loss_verts), filename=f'diffmcubes#{i}.ply')
