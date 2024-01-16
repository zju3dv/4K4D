# Feature Cloud Sequence utilities
# This files builds the components for the feature cloud sequence sampler

import torch
from typing import List, Dict, Union

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.raster_utils import get_ndc_perspective_matrix
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.math_utils import normalize_sum, affine_inverse, affine_padding
from easyvolcap.utils.net_utils import MLP


def estimate_occupancy_field(xyz: torch.Tensor, rad: torch.Tensor, occ: torch.Tensor):
    # This method builds a function to evaluate the occupancy field of the point cloud density field
    # We sample the point cloud with a ball query for the largest radius in the set
    # The actual alpha is decreased as the distance to the closest points
    # If multiple points fall into the region of interest, we compute for alpha on all of them and performs a add operation
    from pytorch3d.ops import ball_query
    max_rad = rad.max()
    # B, N, 3
    # B, N, 1
    # B, N, 1

    def field(pts: torch.Tensor, K=10):
        # pts: B, P, 3
        sh = pts.shape
        pts = pts.view(pts.shape[0], -1, 3)
        knn = ball_query(pts, xyz, K=K, radius=max_rad, return_nn=False)
        idx, dists = knn.idx, knn.dists  # B, P, K
        msk = idx != -1
        idx = torch.where(msk, idx, 0).long()
        pix_rad = multi_gather(rad[..., 0], idx.view(idx.shape[0], -1), dim=-1).view(idx.shape)  # B, P, K
        pix_occ = multi_gather(occ[..., 0], idx.view(idx.shape[0], -1), dim=-1).view(idx.shape)  # B, P, K
        pix_occ = pix_occ * (1 - dists / (pix_rad * pix_rad))  # B, P, K
        pix_occ = torch.where(msk, pix_occ, 0)
        pix_occ = pix_occ.clip(0, 1)
        pix_occ = pix_occ.sum(dim=-1, keepdim=True)  # B, P, 1
        return pix_occ.view(*sh[:-1], 1)

    return field

# @torch.jit.script


def prepare_feedback_transform(H: int, W: int, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                               n: torch.Tensor,
                               f: torch.Tensor,
                               xyz: torch.Tensor,
                               rgb: torch.Tensor,
                               rad: torch.Tensor):
    ixt = get_ndc_perspective_matrix(K, H, W, n[..., 0], f[..., 0]).to(xyz.dtype)  # to opengl, remove last dim of n and f
    w2c = affine_padding(torch.cat([R, T], dim=-1)).to(xyz.dtype)
    c2w = affine_inverse(w2c)
    c2w[..., 0] *= 1  # flip x
    c2w[..., 1] *= -1  # flip y
    c2w[..., 2] *= -1  # flip z
    ext = affine_inverse(c2w)
    pix_xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) @ ext.mT @ ixt.mT
    pix_rad = abs(H * ixt[..., 1, 1][..., None, None] * rad / pix_xyz[..., -1:])  # z: B, 1 * B, N, world space radius -> ndc radius B, N, 1

    # Prepare data to be rendered
    data = torch.cat([pix_xyz, rgb, pix_rad], dim=-1).ravel()  # organize the data inside vbo
    return data


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


@run_once
def warn_once_about_pulsar_fxfy():
    log(yellow(
        "Pulsar only supports a single focal lengths. For converting OpenCV "
        "focal lengths, we average them for x and y directions. "
        "The focal lengths for x and y you provided differ by more than 1%, "
        "which means this could introduce a noticeable error."
    ))


def get_pulsar_camera_params(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
    znear: float = 0.1,
) -> torch.Tensor:
    assert len(camera_matrix.size()) == 3, "This function requires batched inputs!"
    assert len(R.size()) == 3, "This function requires batched inputs!"
    assert len(tvec.size()) in (2, 3), "This function reuqires batched inputs!"

    # Validate parameters.
    image_size_wh = image_size.to(R).flip(dims=(1,))
    assert torch.all(
        image_size_wh > 0
    ), "height and width must be positive but min is: %s" % (
        str(image_size_wh.min().item())
    )
    assert (
        camera_matrix.size(1) == 3 and camera_matrix.size(2) == 3
    ), "Incorrect camera matrix shape: expected 3x3 but got %dx%d" % (
        camera_matrix.size(1),
        camera_matrix.size(2),
    )
    assert (
        R.size(1) == 3 and R.size(2) == 3
    ), "Incorrect R shape: expected 3x3 but got %dx%d" % (
        R.size(1),
        R.size(2),
    )
    if len(tvec.size()) == 2:
        tvec = tvec.unsqueeze(2)
    assert (
        tvec.size(1) == 3 and tvec.size(2) == 1
    ), "Incorrect tvec shape: expected 3x1 but got %dx%d" % (
        tvec.size(1),
        tvec.size(2),
    )
    # Check batch size.
    batch_size = camera_matrix.size(0)
    assert R.size(0) == batch_size, "Expected R to have batch size %d. Has size %d." % (
        batch_size,
        R.size(0),
    )
    assert (
        tvec.size(0) == batch_size
    ), "Expected tvec to have batch size %d. Has size %d." % (
        batch_size,
        tvec.size(0),
    )
    # Check image sizes.
    image_w = image_size_wh[0, 0]
    image_h = image_size_wh[0, 1]
    assert torch.all(
        image_size_wh[:, 0] == image_w
    ), "All images in a batch must have the same width!"
    assert torch.all(
        image_size_wh[:, 1] == image_h
    ), "All images in a batch must have the same height!"
    # Focal length.
    fx = camera_matrix[:, 0, 0].unsqueeze(1)
    fy = camera_matrix[:, 1, 1].unsqueeze(1)
    # Check that we introduce less than 1% error by averaging the focal lengths.
    fx_y = fx / fy
    if torch.any(fx_y > 1.01) or torch.any(fx_y < 0.99):
        warn_once_about_pulsar_fxfy()
    f = (fx + fy) / 2
    # Normalize f into normalized device coordinates.
    focal_length_px = f / image_w
    # Transfer into focal_length and sensor_width.
    # NOTE: Using torch.tensor instead of torch.as_tensor will cause cpu gpu sync
    focal_length = torch.as_tensor([znear - 1e-5], dtype=torch.float32, device=R.device)
    focal_length = focal_length[None, :].repeat(batch_size, 1)
    sensor_width = focal_length / focal_length_px
    # Principal point.
    cx = camera_matrix[:, 0, 2].unsqueeze(1)
    cy = camera_matrix[:, 1, 2].unsqueeze(1)
    # Transfer principal point offset into centered offset.
    cx = -(cx - image_w / 2)
    cy = cy - image_h / 2
    # Concatenate to final vector.
    param = torch.cat([focal_length, sensor_width, cx, cy], dim=1)
    R_trans = R.permute(0, 2, 1)
    cam_pos = -torch.bmm(R_trans, tvec).squeeze(2)
    cam_rot = matrix_to_rotation_6d(R_trans)
    cam_params = torch.cat([cam_pos, cam_rot, param], dim=1)
    return cam_params


def get_opencv_camera_params(batch: dotdict):
    H = batch.meta.H[0].item()  # !: BATCH
    W = batch.meta.W[0].item()  # !: BATCH
    K = batch.K
    R = batch.R
    T = batch.T
    C = -batch.R.mT @ batch.T  # B, 3, 1
    return H, W, K, R, T, C


def get_pytorch3d_camera_params(batch: dotdict):
    # Extract pytorc3d camera parameters from batch input
    # R and T are applied on the right (requires a transposed R from OpenCV camera format)
    # Coordinate system is different from that of OpenCV (cv: right down front, 3d: left up front)
    # However, the correction has to be down on both T and R... (instead of just R)
    C = -batch.R.mT @ batch.T  # B, 3, 1
    R = batch.R.clone()
    R[..., 0, :] *= -1  # flip x row
    R[..., 1, :] *= -1  # flip y row
    T = (-R @ C)[..., 0]  # c2w back to w2c
    R = R.mT  # applied left (left multiply to right multiply, god knows why...)

    H = batch.meta.H[0].item()  # !: BATCH
    W = batch.meta.W[0].item()  # !: BATCH
    K = get_pytorch3d_ndc_K(batch.K, H, W)

    return H, W, K, R, T, C

# TODO: Remove pcd_t and with_t semantics, this is a legacy API


def voxel_surface_down_sample(pcd: torch.Tensor, pcd_t: torch.Tensor = None, voxel_size: float = 0.01, dist_th: float = 0.025, n_points: int = 65536):
    # !: BATCH
    # TODO: Use number of vertices for good estimation
    import open3d as o3d
    import numpy as np
    import mcubes
    from easyvolcap.utils.sample_utils import point_mesh_distance
    from pytorch3d.ops import knn_points, ball_query, sample_farthest_points

    # Convert torch tensor to Open3D PointCloud
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())

    # Create VoxelGrid from PointCloud
    o3d_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(o3d_pcd, voxel_size=voxel_size)

    # Extract dense grid from VoxelGrid using get_voxel
    voxels = o3d_vox.get_voxels()
    max_index = np.array([vox.grid_index for vox in voxels]).max(axis=0)  # !: for-loop
    dense_grid = np.zeros((max_index[0] + 1, max_index[1] + 1, max_index[2] + 1))

    for vox in voxels:  # !: for-loop
        dense_grid[vox.grid_index[0], vox.grid_index[1], vox.grid_index[2]] = 1

    # Use marching cubes to obtain mesh from dense grid
    vertices, triangles = mcubes.marching_cubes(dense_grid, 0.5)
    vertices = vertices * voxel_size + o3d_vox.origin  # resizing

    # Convert mesh data to torch tensors
    triangles_torch = torch.as_tensor(vertices[triangles], device=pcd.device, dtype=pcd.dtype).float()

    # Calculate distances using point_mesh_distance
    dists, _ = point_mesh_distance(pcd[0], triangles_torch)

    # Select points based on distances
    valid = (dists < dist_th).nonzero()[..., 0]
    while (len(valid) - n_points) / n_points > 0.005:
        # There are too many valid points, should control its number
        ratio = len(valid) / len(pcd[0])  # the ratio of valid points
        n_expected = int(n_points / ratio)  # the expected number of points before surface sampling
        pcd = random(pcd, n_points=n_expected)

        # Calculate distances using point_mesh_distance
        dists, _ = point_mesh_distance(pcd[0], triangles_torch)

        # Select points based on distances
        valid = (dists < dist_th).nonzero()[..., 0]

    _, valid = dists.topk(n_points, dim=-1, sorted=False, largest=False)
    pcd_new = torch.index_select(pcd[0], 0, valid)[None]

    return pcd_new


def filter_bounds(pcd: torch.Tensor, pcd_t: torch.Tensor = None, bounds: torch.Tensor = None):
    valid = ((pcd - bounds[..., 0, :]) > 0).all(dim=-1) & ((pcd - bounds[..., 1, :]) < 0).all(dim=-1)  # mask: B, N
    valid = valid[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC
    pcd = multi_gather(pcd, valid, dim=-2)
    return pcd


def duplicate(pcd: torch.Tensor, pcd_t: torch.Tensor = None, std: float = 0.005 * 0.1):
    # return pcd.repeat_interleave(2, dim=-2), ind.repeat_interleave(2, dim=-2)
    pcd_new = torch.normal(pcd, std=std)
    return torch.cat([pcd, pcd_new], dim=-2)


def farthest(pcd: torch.Tensor, pcd_t: torch.Tensor = None, lengths: torch.Tensor = None, n_points: int = 65536):
    from pytorch3d.ops import knn_points, ball_query, sample_farthest_points
    idx = sample_farthest_points(pcd, lengths, K=n_points)[1]  # N, K (padded)
    return multi_gather(pcd, idx)


def random(pcd: torch.Tensor, pcd_t: torch.Tensor = None, n_points: int = 65536, std: float = 0.001):
    inds = torch.stack([torch.randperm(pcd.shape[-2], device=pcd.device)[:n_points] for b in range(len(pcd))])  # B, S,
    return multi_gather(pcd, inds)


def voxel_down_sample(pcd: torch.Tensor, pcd_t: torch.Tensor = None, voxel_size=0.005):
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
    return torch.as_tensor(np.array(o3d_pcd.points)).to(pcd.device, pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def remove_outlier(pcd: torch.Tensor, pcd_t: torch.Tensor = None, K: int = 20, std_ratio=2.0, return_inds=False):  # !: BATCH
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=K, std_ratio=std_ratio)
    if return_inds:
        return torch.as_tensor(np.array(ind), device=pcd.device)[None]  # N,
    return torch.as_tensor(np.array(o3d_pcd.points)[np.array(ind)]).to(pcd.device, pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def farthest_down_sample(pcd: torch.Tensor, pcd_t: torch.Tensor = None, K: int = 65536):
    import open3d as o3d
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.view(-1, 3).detach().cpu().numpy())
    o3d_pcd = o3d_pcd.farthest_point_down_sample(K)
    return torch.as_tensor(np.array(o3d_pcd.points)).to(pcd.device, pcd.dtype, non_blocking=True).view(pcd.shape[0], -1, 3)


def sample_random_points(pcd: torch.Tensor, pcd_t: torch.Tensor = None, K: int = 500):
    bounds = torch.stack([pcd.min(dim=-2)[0] - 0.033, pcd.max(dim=-2)[0] + 0.033], dim=-2)  # B, 2, 3
    pts = torch.rand(*pcd.shape[:-2], K, 3, device=pcd.device) * (bounds[..., 1:, :] - bounds[..., :1, :]) + bounds[..., :1, :]
    return pts


def sample_filter_random_points(pcd: torch.Tensor, pcd_t: torch.Tensor = None, K: int = 500, update_radius=0.05, filter_K=10):
    pts = sample_random_points(pcd, pcd_t, K)  # ugly interface
    pts = filter_points(pts, pcd, update_radius, filter_K)
    return pts


def get_pytorch3d_ndc_K(K: torch.Tensor, H: int, W: int):
    M = min(H, W)
    K = torch.cat([K, torch.zeros_like(K[..., -1:, :])], dim=-2)
    K = torch.cat([K, torch.zeros_like(K[..., :, -1:])], dim=-1)
    K[..., 3, 2] = 1  # ...? # HACK: pytorch3d magic
    K[..., 2, 2] = 0  # ...? # HACK: pytorch3d magic
    K[..., 2, 3] = 1  # ...? # HACK: pytorch3d magic

    K[..., 0, 1] = 0
    K[..., 1, 0] = 0
    K[..., 2, 0] = 0
    K[..., 2, 1] = 0
    # return K

    K[..., 0, 0] = K[..., 0, 0] * 2.0 / M  # fx
    K[..., 1, 1] = K[..., 1, 1] * 2.0 / M  # fy
    K[..., 0, 2] = -(K[..., 0, 2] - W / 2.0) * 2.0 / M  # px
    K[..., 1, 2] = -(K[..., 1, 2] - H / 2.0) * 2.0 / M  # py
    return K


def expand_points_features(render_scale: Union[float, int], pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float):
    # FIXME: Duplicated code for these
    n_points = pcd_old.shape[-2]
    if isinstance(render_scale, int):
        target_n_points = render_scale
        n_points = pcd_old.shape[-2]
        render_scale = target_n_points / n_points
    target_n_points = int(render_scale * n_points)
    return generate_points_features(target_n_points, pcd_old, ind_old, radius)


def expand_points(render_scale: Union[float, int], pcd_old: torch.Tensor, radius: float):
    n_points = pcd_old.shape[-2]
    if isinstance(render_scale, int):
        target_n_points = render_scale
        n_points = pcd_old.shape[-2]
        render_scale = target_n_points / n_points
    target_n_points = int(render_scale * n_points)
    return generate_points(target_n_points, pcd_old, radius)


def generate_points_features(n_points: int, pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float):
    pcd_new = sample_random_points(pcd_old, K=n_points)
    pcd_new, ind_new = update_points_features(pcd_new, pcd_old, ind_old, radius)
    return pcd_new, ind_new


def generate_points(n_points: int, pcd_old: torch.Tensor, radius: float):
    pcd_new = sample_random_points(pcd_old, K=n_points)
    pcd_new = update_points(pcd_new, pcd_old, radius)
    return pcd_new


def surface_points(pcd: torch.Tensor, pcd_t: torch.Tensor = None, radius: float = 0.05, K: int = 500, n_points: float = 16384):
    # Try to retain the surface points
    from pytorch3d.ops import knn_points, ball_query

    # 1. Perform a ball query (with a large upper limit number of points)
    # 2. Sort all points based on the number of neighbors
    close = ball_query(pcd, pcd, radius=radius, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx

    dists = torch.where(idx == -1, torch.inf, 0.1)  # B, S, K, equal weight, just for filtering
    idx = torch.where(idx == -1, 0, idx)  # B, S, K

    # Find mean points
    B, S, C = pcd.shape
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = multi_gather(pcd, idx.view(B, S * K)).view(B, S, K, -1)
    pcd_new = (pcd_new * weights).sum(dim=-2)  # B, S, 3

    # Find mean deviation
    dists = (pcd_new - pcd).norm(dim=-1)  # B, S,
    valid = (dists).topk(n_points, dim=-1, sorted=False)[1]  # B, K
    pcd_new = multi_gather(pcd, valid, dim=-2)

    return pcd_new


def surface_points_features(pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float = 0.05, K: int = 500, n_points: float = 16384):
    # Try to retain the surface points
    from pytorch3d.ops import knn_points, ball_query

    # 1. Perform a ball query (with a large upper limit number of points)
    # 2. Sort all points based on the number of neighbors
    close = ball_query(pcd_old, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx

    dists = torch.where(idx == -1, torch.inf, 0.1)  # B, S, K, equal weight, just for filtering
    idx = torch.where(idx == -1, 0, idx)  # B, S, K

    # Find mean points
    B, S, C = pcd_old.shape
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)
    pcd_new = (pcd_new * weights).sum(dim=-2)  # B, S, 3

    # Find mean deviation
    dists = (pcd_new - pcd_old).norm(dim=-1)  # B, S,
    valid = (dists).topk(n_points, dim=-1, sorted=False)[1]  # B, K
    pcd_new = multi_gather(pcd_old, valid, dim=-2)
    ind_new = multi_gather(ind_old, valid, dim=-2)

    return pcd_new, ind_new


def filter_points(pcd_new: torch.Tensor, pcd_old: torch.Tensor, radius: float = 0.05, K: int = 10, fill_ratio: float = 0.1):
    # This will lead to shrinking
    from pytorch3d.ops import knn_points, ball_query

    close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx
    # !: BATCH
    good = (idx != -1).sum(dim=-1) / K > fill_ratio
    valid = good[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC

    idx = multi_gather(idx, valid, dim=-2)
    dists = multi_gather(dists, valid, dim=-2)
    pcd_new = multi_gather(pcd_new, valid, dim=-2)
    dists = torch.where(idx == -1, torch.inf, dists)  # B, S, K
    idx = torch.where(idx == -1, 0, idx)  # B, S, K

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    return pcd_new


def filter_points_features(pcd_new: torch.Tensor, pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float = 0.05, K: int = 10, fill_ratio: float = 0.1):
    # This will lead to shrinking
    from pytorch3d.ops import knn_points, ball_query

    close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx
    # !: BATCH
    good = (idx != -1).sum(dim=-1) / K > fill_ratio
    valid = good[0].nonzero()[None]  # B, S -> B, V # MARK: SYNC

    idx = multi_gather(idx, valid, dim=-2)
    dists = multi_gather(dists, valid, dim=-2)
    pcd_new = multi_gather(pcd_new, valid, dim=-2)
    dists = torch.where(idx == -1, torch.inf, dists)  # B, S, K
    idx = torch.where(idx == -1, 0, idx)  # B, S, K

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    ind_new = (ind_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return pcd_new, ind_new


def update_points_features(pcd_new: torch.Tensor, pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = knn_points(pcd_new, pcd_old, return_sorted=False, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    ind_new = (ind_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return pcd_new, ind_new


def update_points(pcd_new: torch.Tensor, pcd_old: torch.Tensor, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = knn_points(pcd_new, pcd_old, return_sorted=False, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    pcd_new = multi_gather(pcd_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    pcd_new = (pcd_new * weights).sum(dim=-2)
    # pcd_new = pcd_new.mean(dim=-2)
    return pcd_new


def update_features(pcd_new: torch.Tensor, pcd_old: torch.Tensor, ind_old: torch.Tensor, radius: float = 0.05, K: int = 5):
    # This will lead to shrinking
    from pytorch3d.ops import knn_points, ball_query

    # close = ball_query(pcd_new, pcd_old, radius=radius, return_nn=False, K=K)  # B, S, K
    close = knn_points(pcd_new, pcd_old, return_sorted=False, return_nn=False, K=K)  # B, S, K
    dists, idx = close.dists, close.idx

    B, S, C = pcd_new.shape
    B, N, C = pcd_old.shape
    ind_new = multi_gather(ind_old, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, C
    weights = weight_function(dists, radius)[..., None]  # B, S, K, 1
    ind_new = (ind_new * weights).sum(dim=-2)
    # ind_new = ind_new.mean(dim=-2)
    return ind_new


def weight_function(d2: torch.Tensor, radius: float = 0.05, delta: float = 0.001):
    # Radius weighted function from structured local radiance field
    weights = (-d2 / (2 * radius ** 2)).exp().clip(0)  # B, S, K
    weights = normalize_sum(weights)
    return weights
