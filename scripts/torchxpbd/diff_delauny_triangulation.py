# NOTE: under development now...
# FIXME: not working yet

# will:
# 1. sample random points on this mesh
# 2. find corresponding surface points of this mesh by sphere tracing along normal direction (skipped)
# 3. perform delauny triangulation to form a new mesh
# 4. extract surface porperties from the sdf points, semantics, normals, uvs, etc. (maybe even high level feature?) (skipped)
# 5. perform forward deformation through a residual deformation network (skipped)
# 6. perform physics loss to regularize the deformation field

import torch
from tqdm import tqdm
from torch.optim import Adam
from pytorch3d.structures import Meshes
from scipy.spatial import Delaunay

# fmt: off
import sys

sys.path.append('.')
from lib.networks.deform.base_network import SignedDistanceNetwork
from lib.networks.deform.base_network import ResidualDeformation
from lib.networks.deform.bijective_network import BijectiveDeformation

from easyvolcap.utils.net_utils import get_bounds, make_params, normalize, take_gradient
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.loss_utils import eikonal, l2_reg
from easyvolcap.utils.mesh_utils import laplacian_smoothing, differentiable_marching_cubes, register_sdf_gradient
from easyvolcap.utils.data_utils import export_mesh, load_mesh
from easyvolcap.utils.sample_utils import random_points_on_meshes_with_face_and_bary
# fmt: on

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

lr = 1e-3
iter_cnt = 500
num_samples = 10000
optim = Adam(sdf.parameters(), lr=lr)
for i in tqdm(range(iter_cnt)):
    mesh = Meshes([verts.detach()], [faces.detach()])
    verts = random_points_on_meshes_with_face_and_bary(mesh, 10000)[0][0].requires_grad_()
    
    # !: how to acquire surface triangulation?
    faces = Delaunay(verts.detach().cpu().numpy(), qhull_options='QJ').simplices # this is a 4D triangulation...
    __import__('ipdb').set_trace()
    faces = torch.from_numpy(faces).to(verts.device, torch.long, non_blocking=True)

    # grad start
    vsdf = sdf.sdf(verts)
    grad = take_gradient(vsdf, verts)
    grad_loss = eikonal(grad)

    verts = register_sdf_gradient(verts, sdf.sdf)
    data_loss_verts = (verts.norm(dim=-1) - 0.5).abs()
    data_loss = data_loss_verts.mean()
    loss = data_loss + 0.1 * grad_loss
    # grad end

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

    vsdf_mean = vsdf.abs().mean().item()
    log(f'iter: {i}')
    log(f'data_loss: {data_loss.item()}')
    log(f'grad_loss: {grad_loss.item()}')
    log(f'vsdf_mean: {vsdf_mean}', 'red' if vsdf_mean > 1e-4 else 'green')

    export_mesh(verts, faces, colors=colormap(data_loss_verts), filename=f'diffmcubes#{i}.ply')
