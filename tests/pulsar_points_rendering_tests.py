from pytorch3d.renderer.points.pulsar import Renderer as Pulsar
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.utils import pulsar_from_cameras_projection
from pytorch3d.renderer.points.rasterizer import PointFragments, rasterize_points
from pytorch3d.io import load_ply
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    PulsarPointsRenderer
)
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.fcds_utils import get_pytorch3d_camera_params, get_pulsar_camera_params
from easyvolcap.utils.data_utils import save_image, to_cuda, add_batch
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.test_utils import my_tests
import torch
from copy import deepcopy, copy

filename = 'assets/meshes/bunny.ply'
radius = 0.0025  # 5mm point size
WIDTH, HEIGHT = 512, 512
camera = Camera(H=HEIGHT, W=WIDTH,
                K=torch.tensor([[592., 0., 256.],
                                [0., 592., 256.],
                                [0., 0., 1.]]),
                R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                [-0.1341, -0.9815, -0.1365],
                                [0.0185, 0.1353, -0.9906]]),
                T=torch.tensor([[0.0178],
                                [0.0953],
                                [0.3137]]))
verts, faces = to_cuda(load_ply(filename))
pcd = verts
mesh = Meshes([verts], [faces])
normals = mesh.verts_normals_packed()
feat = normals * 0.5 + 0.5  # render the color out
pcd, feat = add_batch([pcd, feat])
radius = pcd.new_full((*pcd.shape[:-1], 1), radius)
pts_per_pix = 15

# Prepare the pulsa renderer
pulsar = Pulsar(WIDTH, HEIGHT, max_num_balls=1048576, n_channels=5, n_track=pts_per_pix, right_handed_system=False).to('cuda', non_blocking=True)

# Prepare the pytorch3d renderer
rasterizer = PointsRasterizer()
compositor = AlphaCompositor()


def test_pytorch3d_rendering(pcd=pcd, feat=feat, radius=radius):
    H, W, K, R, T, C = get_pytorch3d_camera_params(add_batch(to_cuda(camera.to_batch())))
    cameras = PerspectiveCameras(device='cuda', R=R, T=T, K=K)
    ndc_pcd = rasterizer.transform(Pointclouds(pcd), cameras=cameras).points_padded()  # B, N, 3
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor
    radius = abs(K[..., 1, 1][..., None] * radius / (ndc_pcd[..., -1] + 1e-10))  # z: B, 1 * B, N, world space radius

    # Actual forward rasterization
    # FIXME: STUPID PYTORCH3D POINT CLOUD CREATION WILL SYNC CUDA
    idx, zbuf, dists = rasterize_points(Pointclouds(ndc_pcd), (H, W), radius, pts_per_pix, None, None)
    frags = PointFragments(idx=idx, zbuf=zbuf, dists=dists)
    idx, zbuf, dists = frags.idx, frags.zbuf, frags.dists

    # Prepare for composition
    pix_radius = multi_gather(radius, torch.where(idx == -1, 0, idx).view(radius.shape[0], -1).long(), dim=-1).view(idx.shape)  # B, H, W, K (B, HWK -> B, N -> B, H, W, K)
    pix_weight = 1 - dists / (pix_radius * pix_radius)  # B, H, W, K
    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = (pcd - C.mT).norm(dim=-1, keepdim=True)
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C

    # The actual computation
    feat = compositor(idx.long().permute(0, 3, 1, 2),
                      pix_weight.permute(0, 3, 1, 2),
                      feat.view(-1, feat.shape[-1]).permute(1, 0)).permute(0, 2, 3, 1)  # B, H, W, 3

    # TODO: Implement and return random background here
    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_pytorch3d_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_pytorch3d_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_pytorch3d_rendering_acc.png', acc_map[0].detach().cpu().numpy())


def test_hybrid_pulsar_rendering(pcd=pcd, feat=feat, radius=radius):
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor

    H, W, K, R, T, C = get_pytorch3d_camera_params(add_batch(to_cuda(camera.to_batch())))
    cameras = PerspectiveCameras(device='cuda', R=R, T=T, K=K)
    cam_params = pulsar_from_cameras_projection(cameras, torch.as_tensor([HEIGHT, WIDTH], device=pcd.device)[None])

    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = (pcd - C.mT).norm(dim=-1, keepdim=True)
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C

    # NOTE: DOUBLE CHECK MAX DEPTH USAGE
    feat = pulsar(pcd, feat, radius, cam_params, 5e-5, 10.0, 0.0, torch.zeros(5, device=pcd.device)).flip(1)

    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_hybrid_pulsar_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_hybrid_pulsar_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_hybrid_pulsar_rendering_acc.png', acc_map[0].detach().cpu().numpy())


def test_manual_pulsar_rendering(pcd=pcd, feat=feat, radius=radius):
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor
    batch = add_batch(to_cuda(camera.to_batch()))
    K, R, T = batch.K, batch.R, batch.T
    C = -R.mT @ T

    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = (pcd - C.mT).norm(dim=-1, keepdim=True)  # world space depth
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C

    pcd = pcd @ R.mT + T.mT  # apply w2c conversion

    T = torch.zeros_like(T)
    R = torch.zeros_like(R)
    R[..., torch.arange(3), torch.arange(3)] = 1.0  # identity
    cam_params = get_pulsar_camera_params(R, T[..., 0], K, torch.as_tensor([HEIGHT, WIDTH], device=pcd.device)[None], batch.meta.n.item())

    # NOTE: DOUBLE CHECK MAX DEPTH USAGE
    feat = pulsar(pcd, feat, radius, cam_params, 5e-5, batch.meta.f.item(), batch.meta.n.item(), torch.zeros(5, device=pcd.device)).flip(1)

    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_manual_pulsar_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_manual_pulsar_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_manual_pulsar_rendering_acc.png', acc_map[0].detach().cpu().numpy())


def test_pulsar_rendering(pcd=pcd, feat=feat, radius=radius):
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor

    batch = add_batch(to_cuda(camera.to_batch()))
    K, R, T = batch.K, batch.R, batch.T
    C = -R.mT @ T
    front = R[..., -1:, :]  # B, 1, 3

    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = ((pcd - C.mT) * front).sum(dim=-1, keepdims=True)  # the rounding depth, not parallel to the axis of projection
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C
    depth_max, depth_min = depth.max().item() + 0.005, depth.min().item() - 0.005

    cam_params = get_pulsar_camera_params(R, T[..., 0], K, torch.as_tensor([HEIGHT, WIDTH], device=pcd.device)[None], depth_min)  # strange looking camera, # NOTE: no effect on final image

    # NOTE: DOUBLE CHECK MAX DEPTH USAGE
    feat = pulsar(pcd, feat, radius, cam_params, 1e-4, depth_max, depth_min, torch.zeros(5, device=pcd.device)).flip(1)  # sharper? a little bit? # FIXME: different looks

    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_pulsar_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_pulsar_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_pulsar_rendering_acc.png', acc_map[0].detach().cpu().numpy())


def test_resized_pytorch3d_rendering(pcd=pcd, feat=feat, radius=radius):
    camera = Camera(H=128, W=256,
                    K=torch.tensor([[296., 0., 128.],
                                    [0., 296., 64.],
                                    [0., 0., 1.]]),
                    R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                    [-0.1341, -0.9815, -0.1365],
                                    [0.0185, 0.1353, -0.9906]]),
                    T=torch.tensor([[0.0178],
                                    [0.0953],
                                    [0.3137]]))

    H, W, K, R, T, C = get_pytorch3d_camera_params(add_batch(to_cuda(camera.to_batch())))
    cameras = PerspectiveCameras(device='cuda', R=R, T=T, K=K)
    ndc_pcd = rasterizer.transform(Pointclouds(pcd), cameras=cameras).points_padded()  # B, N, 3
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor
    radius = abs(K[..., 1, 1][..., None] * radius / (ndc_pcd[..., -1] + 1e-10))  # z: B, 1 * B, N, world space radius

    # Actual forward rasterization
    # FIXME: STUPID PYTORCH3D POINT CLOUD CREATION WILL SYNC CUDA
    idx, zbuf, dists = rasterize_points(Pointclouds(ndc_pcd), (H, W), radius, pts_per_pix, None, None)
    frags = PointFragments(idx=idx, zbuf=zbuf, dists=dists)
    idx, zbuf, dists = frags.idx, frags.zbuf, frags.dists

    # Prepare for composition
    pix_radius = multi_gather(radius, torch.where(idx == -1, 0, idx).view(radius.shape[0], -1).long(), dim=-1).view(idx.shape)  # B, H, W, K (B, HWK -> B, N -> B, H, W, K)
    pix_weight = 1 - dists / (pix_radius * pix_radius)  # B, H, W, K
    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = (pcd - C.mT).norm(dim=-1, keepdim=True)
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C

    # The actual computation
    feat = compositor(idx.long().permute(0, 3, 1, 2),
                      pix_weight.permute(0, 3, 1, 2),
                      feat.view(-1, feat.shape[-1]).permute(1, 0)).permute(0, 2, 3, 1)  # B, H, W, 3

    # TODO: Implement and return random background here
    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_resized_pytorch3d_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_resized_pytorch3d_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_resized_pytorch3d_rendering_acc.png', acc_map[0].detach().cpu().numpy())


def test_resized_pulsar_rendering(pcd=pcd, feat=feat, radius=radius):
    camera = Camera(H=128, W=256,
                    K=torch.tensor([[296., 0., 128.],
                                    [0., 296., 64.],
                                    [0., 0., 1.]]),
                    R=torch.tensor([[0.9908, -0.1353, 0.0000],
                                    [-0.1341, -0.9815, -0.1365],
                                    [0.0185, 0.1353, -0.9906]]),
                    T=torch.tensor([[0.0178],
                                    [0.0953],
                                    [0.3137]]))
    H, W = camera.H, camera.W
    if isinstance(radius, torch.Tensor): radius = radius[..., 0]  # remove last dim to make it homogenous for float and tensor

    batch = add_batch(to_cuda(camera.to_batch()))
    K, R, T = batch.K, batch.R, batch.T
    C = -R.mT @ T
    front = R[..., -1:, :]  # B, 1, 3

    acc = torch.ones_like(feat[..., :1])
    depth: torch.Tensor = ((pcd - C.mT) * front).sum(dim=-1, keepdims=True)  # the rounding depth, not parallel to the axis of projection
    feat = torch.cat([feat, acc, depth], dim=-1)  # B, N, 3 + C
    depth_max, depth_min = depth.max().item() + 0.005, depth.min().item() - 0.005

    cam_params = get_pulsar_camera_params(R, T[..., 0], K, torch.as_tensor([HEIGHT, WIDTH], device=pcd.device)[None], depth_min)  # strange looking camera, # NOTE: no effect on final image

    # NOTE: DOUBLE CHECK MAX DEPTH USAGE
    feat = pulsar(pcd, feat, radius, cam_params, 1e-3, depth_max, depth_min, torch.zeros(5, device=pcd.device)).flip(1)  # sharper? a little bit? # FIXME: different looks
    feat = feat[:, :H, :W]

    rgb_map, acc_map, dpt_map = feat[..., :3], feat[..., 3:4], feat[..., 4:5]
    dpt_map = dpt_map + (1 - acc_map) * depth.max()
    rgb_map, acc_map, dpt_map = torch.cat([rgb_map, acc_map], dim=-1), torch.cat([acc_map, acc_map, acc_map, acc_map], dim=-1), torch.cat([dpt_map, dpt_map, dpt_map, acc_map], dim=-1)

    save_image('test_resized_pulsar_rendering_rgb.png', rgb_map[0].detach().cpu().numpy())
    save_image('test_resized_pulsar_rendering_dpt.png', dpt_map[0].detach().cpu().numpy())
    save_image('test_resized_pulsar_rendering_acc.png', acc_map[0].detach().cpu().numpy())


if __name__ == "__main__":
    my_tests(globals())
