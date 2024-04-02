# Adopted from https://github.com/autonomousvision/differentiable_volumetric_rendering/blob/master/im2mesh/dvr/models/depth_function.py
import torch
import torch.nn.functional as F

from torch import nn
from typing import Mapping

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.color_utils import colormap
from easyvolcap.utils.ray_utils import create_meshgrid
from easyvolcap.utils.ibr_utils import depth_regression
from easyvolcap.utils.math_utils import affine_inverse, torch_inverse_3x3


def depth_confidence(depth_prob: torch.Tensor):
    depth_range = torch.arange(depth_prob.shape[-3], dtype=torch.float, device=depth_prob.device)  # ,
    depth_range = depth_range[None, :, None, None].expand(depth_prob.shape)  # B, D, H, W

    depth_index, _ = depth_regression(depth_prob, depth_range)  # B, H, W
    depth_index = depth_index.long()

    depth_prob_sum4 = depth_prob.softmax(dim=-3)[:, None]  # B, 1, D, H, W
    depth_prob_sum4 = F.pad(depth_prob_sum4, pad=(0, 0, 0, 0, 1, 2))  # B, 1, D + 3, H, W
    depth_prob_sum4 = F.avg_pool3d(depth_prob_sum4, (4, 1, 1), stride=1, padding=0)[:, 0]  # B, D, H, W
    depth_prob_sum4 = depth_prob_sum4 * 4  # B, D, H, W

    conf = depth_prob_sum4.gather(-3, depth_index[..., None, :, :]).view(-1)  # B, 1, H, W -> -1

    return conf


def depth2xyz(depth: torch.Tensor, ixts: torch.Tensor, exts: torch.Tensor, correct_pix: bool = True):
    # depth: B, S, 1, H, W or B, S, H, W
    # exts: B, S, 4, 4
    # ixts: B, S, 3, 3

    # Deal with the depth dimension
    if depth.ndim > 4:
        depth = depth[..., 0, :, :]  # remove the channel dimension for shape preparation

    # Prepare shapes
    B, S, H, W = depth.shape
    depth = depth.reshape((-1, ) + depth.shape[-2:])  # (B * S, H, W)
    ixts = ixts.reshape((-1, ) + ixts.shape[-2:])  # (B * S, 3, 3)
    # The input exts is world to camera by default
    c2ws = affine_inverse(exts).reshape((-1, ) + exts.shape[-2:])  # (B * S, 4, 4)

    # Create the meshgrid for source images
    ref_grid = create_meshgrid(H, W, device=depth.device, correct_pix=correct_pix, dtype=depth.dtype).flip(-1)  # (H, W, 2), in ij ordering
    ref_grid = torch.cat([ref_grid, torch.ones_like(ref_grid[..., :1])], dim=-1)  # (H, W, 3)
    ref_grid = ref_grid[None].expand(B * S, -1, -1, -1).permute(0, 3, 1, 2).reshape(B * S, 3, -1)  # (B * S, 3, H * W)

    # Compute the xyz coordinates in the camera space
    cam_xyz = torch_inverse_3x3(ixts) @ ref_grid  # (B * S, 3, H * W)
    cam_xyz = cam_xyz * depth.reshape(B * S, 1, -1)  # (B * S, 3, H * W)

    # Transform the xyz coordinates to the world space
    world_xyz = c2ws[..., :3, :3] @ cam_xyz + c2ws[..., :3, 3:]  # (B * S, 3, H * W)
    world_xyz = world_xyz.reshape(B, S, 3, H, W)  # (B, S, 3, H, W)

    return world_xyz


@torch.jit.script
def normalize_depth(depth: torch.Tensor, p: float = 0.01):
    n = int(depth.numel() * p)
    near = depth.ravel().topk(n, largest=False)[0].max()  # a simple version of percentile
    far = depth.ravel().topk(n, largest=True)[0].min()  # a simple version of percentile
    depth = 1 - (depth - near) / (far - near)
    depth = depth.clip(0, 1)
    return depth


def depth_curve_fn(depth: torch.Tensor, p: float = 0.01, cm: str = 'linear'):
    depth = normalize_depth(depth)
    depth = colormap(depth, cm)
    return depth


def bisection(d_low: torch.Tensor, d_high: torch.Tensor, n_steps: int, ray_o: torch.Tensor, ray_d: torch.Tensor,
              decoder: nn.Module, batch: Mapping[str, torch.Tensor], tau: float):
    ''' Runs the bisection method for interval [d_low, d_high].
    Args:
        d_low: start values for the interval
        d_high: end values for the interval
        n_steps: number of steps
        ray_o: masked ray start points
        ray_d: masked ray direction vectors
        decoder: decoder model to evaluate point occupancies
        batch: other input
        tau: threshold value
    '''
    d_pred = (d_low + d_high) / 2.
    for i in range(n_steps):
        p_mid = ray_o + d_pred.unsqueeze(-1) * ray_d
        with torch.no_grad():
            # !: BATCH
            f_mid = decoder(p_mid[None], batch)[0, :, 0] - tau
        ind_low = f_mid < 0
        d_low[ind_low] = d_pred[ind_low]
        d_high[ind_low == 0] = d_pred[ind_low == 0]
        d_pred = 0.5 * (d_low + d_high)
    return d_pred


def secant(f_low: torch.Tensor, f_high: torch.Tensor, d_low: torch.Tensor, d_high: torch.Tensor,
           n_steps: int, ray_o: torch.Tensor, ray_d: torch.Tensor, decoder: nn.Module,
           batch: Mapping[str, torch.Tensor], tau: float):
    ''' Runs the secant method for interval [d_low, d_high].
    Args:
        f_low: start output value for the interval
        f_low: end output value for the interval
        d_low: start values for the interval
        d_high: end values for the interval
        n_steps: number of steps
        ray_o: masked ray start points
        ray_d: masked ray direction vectors
        decoder: decoder model to evaluate point occupancies
        batch: other input
        tau: threshold value
    '''
    d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_steps):
        p_mid = ray_o + d_pred.unsqueeze(-1) * ray_d
        with torch.no_grad():
            # !: BATCH
            f_mid = decoder(p_mid[None], batch)[0, :, 0] - tau
        ind_low = f_mid < 0
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
    return d_pred


def ray_marching(ray_o: torch.Tensor, ray_d: torch.Tensor, decoder: nn.Module,
                 batch: Mapping[str, torch.Tensor],
                 near: torch.Tensor, far: torch.Tensor,
                 occ_th: float = 0.5, n_coarse_steps: int = 32, n_refine_steps: int = 8,
                 method: str = 'secant', chunk_size: int = 1024 * 64):
    ''' Performs ray marching to detect surface points.
    The function returns the surface points as well as d_i of the formula
        ray(d_i) = ray0 + d_i * ray_direction
    which hit the surface points. In addition, masks are returned for illegal values.
    Args:
        ray_o: ray start points of dimension B x N x 3
        ray_d: ray direction vectors of dim B x N x 3
        decoder: decoder model to evaluate point occupancies
        batch: other network input
        near: B x N,
        far: B x N,
        tau: threshold value
        n_coarse_steps: interval from which the number of evaluation steps if sampled
        n_refine_steps: number of secant refinement steps
        depth_range: range of possible depth values, can be None if check_bound_intersection
        method: refinement method (default: secant)
        max_points: max number of points loaded to GPU memory
    '''

    # Shotscuts
    n_batch, n_rays, D = ray_o.shape
    device = ray_o.device

    # d_proposal are "proposal" depth values and p_proposal the corresponding "proposal" 3D points
    t_vals = torch.linspace(0, 1, steps=n_coarse_steps, device=device).view(1, 1, n_coarse_steps)
    t_vals = t_vals.repeat(n_batch, n_rays, 1)  # (n_batch, n_rays, n_coarse_steps)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals  # (n_batch, n_rays, n_coarse_steps)

    p_vals = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]  # (n_batch, n_points, n_coarse_steps, 1)

    # Evaluate all proposal points in parallel
    with torch.no_grad():
        val = torch.cat(
            [
                (decoder(p_split, batch) - occ_th)
                for p_split in
                torch.split(
                    p_vals.view(n_batch, -1, 3),
                    int(chunk_size / n_batch), dim=1)
            ],
            dim=1).view(n_batch, -1, n_coarse_steps)

    _, indices, mask = get_mask_from_occ(val)

    # Get depth values and function values for the interval
    # to which we want to apply the Secant method
    n = n_batch * n_rays
    d_low = z_vals.view(n, n_coarse_steps, 1)[torch.arange(n), indices.view(n)].view(n_batch, n_rays)[mask]
    f_low = val.view(n, n_coarse_steps, 1)[torch.arange(n), indices.view(n)].view(n_batch, n_rays)[mask]
    indices = torch.clamp(indices + 1, max=n_coarse_steps - 1)
    d_high = z_vals.view(n, n_coarse_steps, 1)[torch.arange(n), indices.view(n)].view(n_batch, n_rays)[mask]
    f_high = val.view(n, n_coarse_steps, 1)[torch.arange(n), indices.view(n)].view(n_batch, n_rays)[mask]

    ray_o = ray_o[mask]
    ray_d = ray_d[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    if method == 'secant' and mask.sum() > 0:
        d_pred = secant(f_low, f_high, d_low, d_high, n_refine_steps, ray_o, ray_d, decoder, batch, occ_th)
    elif method == 'bisection' and mask.sum() > 0:
        d_pred = bisection(d_low, d_high, n_refine_steps, ray_o, ray_d, decoder, batch, occ_th)
    else:
        d_pred = torch.ones(ray_d.shape[0], device=device)

    # for sanity
    d_out = torch.ones(n_batch, n_rays, device=device)
    d_out[mask] = d_pred

    return d_out, mask


def get_mask_from_occ(val: torch.Tensor):
    n_batch, n_rays, n_coarse_steps = val.shape
    device = val.device

    # Create mask for valid points where the first point is not occupied
    mask_0_not_occupied = val[:, :, 0] < 0

    # Calculate if sign change occurred and concat 1 (no sign change) in
    # last dimension
    sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]), torch.ones(n_batch, n_rays, 1, device=device)], dim=-1)
    cost_matrix = sign_matrix * torch.arange(n_coarse_steps, 0, -1, dtype=torch.float32, device=device)
    # Get first sign change and mask for values where a.) a sign changed
    # occurred and b.) no a neg to pos sign change occurred (meaning from
    # inside surface to outside)
    values, indices = torch.min(cost_matrix, -1)
    mask_sign_change = values < 0
    mask_neg_to_pos = val[torch.arange(n_batch), torch.arange(n_rays), indices] < 0

    # Define mask where a valid depth value is found
    mask: torch.Tensor = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

    return values, indices, mask
