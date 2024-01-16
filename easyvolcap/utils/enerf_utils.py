import torch

from torch import nn
from typing import Union, List, Tuple
from torch.nn import functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

# !: IMPORT
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import get_function
from easyvolcap.utils.ray_utils import create_meshgrid
from easyvolcap.utils.image_utils import interpolate_image
from easyvolcap.utils.math_utils import affine_inverse, normalize
from easyvolcap.utils.data_utils import export_pts, export_pcd, export_mesh
from easyvolcap.utils.prop_utils import s_vals_to_z_vals, weighted_percentile
from easyvolcap.utils.nerf_utils import linear_sampling, ray2xyz, volume_rendering, raw2alpha, compute_dist, render_weights


class TruncatedExponential(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, min=-15, max=15))


trunc_exp = TruncatedExponential.apply
def trunc_exp_actvn(x: torch.Tensor): return trunc_exp(x - 1)


def render_debug_cost_volume(ray_o, ray_d, t, z_vals,
                             B, H, W, Ht, Wt, Hs, Ws,
                             proj_mats, cost_vol=None, feat_vol=None, batch=None
                             ):
    ray_o_c = interpolate_image(ray_o.view(B, H, W, 3).permute(0, 3, 1, 2), size=(Ht, Wt)).permute(0, 2, 3, 1).view(B, -1, 3)  # ? INTERP
    ray_d_c = interpolate_image(ray_d.view(B, H, W, 3).permute(0, 3, 1, 2), size=(Ht, Wt)).permute(0, 2, 3, 1).view(B, -1, 3)  # ? INTERP
    z_vals_c = z_vals.permute(0, 2, 3, 1).view(B, Ht * Wt, -1)
    t_c = interpolate_image(t.view(B, H, W, 1).permute(0, 3, 1, 2), size=(Ht, Wt)).permute(0, 2, 3, 1).view(B, -1, 1)  # ? INTERP
    xyz_c, dir_c, t_c, dist_c = ray2xyz(ray_o_c, ray_d_c, t_c, z_vals_c)  # B, H * W * D, 3

    if batch is not None:
        src_inps_s = interpolate_image(batch.src_inps, size=(Hs, Ws))
        for rgb_i in range(src_inps_s.shape[1]):
            rgb_volume = build_cost_vol(z_vals, src_inps_s, proj_mats, agg_method='noop')
            rgb_volume = rgb_volume[:, rgb_i].permute(0, 3, 4, 2, 1).reshape(B, -1, 3)  # B, 3, D, H, W -> B, D * H * W, 3
            export_pts(xyz_c, color=rgb_volume, filename=f'rgb_volume_{rgb_i}.ply')
        rgb_volume = build_cost_vol(z_vals, src_inps_s, proj_mats, agg_method='mean').permute(0, 3, 4, 2, 1).reshape(B, -1, 3)  # B, 3, D, H, W -> B, D * H * W, 3
        export_pts(xyz_c, color=rgb_volume, filename=f'rgb_volume_mean.ply')

    if cost_vol is not None:
        cost_vol_c = cost_vol[:, :3].permute(0, 3, 4, 2, 1).reshape(B, -1, 3)  # B, 3, D, H, W -> B, D * H * W, 3
        export_pts(xyz_c, color=cost_vol_c, filename=f'cost_vol.ply')

    if feat_vol is not None:
        feat_vol_c = feat_vol[:, :3].permute(0, 3, 4, 2, 1).reshape(B, -1, 3)  # B, 3, D, H, W -> B, D * H * W, 3
        export_pts(xyz_c, color=feat_vol_c, filename=f'feat_vol.ply')

    # __import__('easyvolcap.utils.console_utils', fromlist=['debugger']).debugger()
    # import ipdb
    # ipdb.set_trace()
    # breakpoint()


@torch.jit.script
def sample_geometry_feature_image(xyz: torch.Tensor,
                                  src_feat_rgb: torch.Tensor,
                                  src_exts: torch.Tensor,
                                  src_ixts: torch.Tensor,
                                  src_scale: torch.Tensor,
                                  padding_mode: str = 'border',

                                  #   sample_msk: bool = False,
                                  #   src_size: torch.Tensor = None,  # S, 2
                                  ):
    # xyz: B, P, 3
    # src_feat_rgb: B, S, C, Hs, Ws
    B, S, C, Hs, Ws = src_feat_rgb.shape
    B, P, _ = xyz.shape
    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., -1:])], dim=-1)  # homogeneous coordinates

    src_ixts = src_ixts.clone()
    src_ixts[..., :2, :] *= src_scale

    # B, P, 4 -> B, 1, P, 4
    # B, S, 4, 4 -> B, S, 4, 4
    # -> B, S, P, 4
    xyz1 = (xyz1[..., None, :, :] @ src_exts.mT)
    xyzs = xyz1[..., :3] @ src_ixts.mT  # B, S, P, 3 @ B, S, 3, 3
    xy = xyzs[..., :-1] / (xyzs[..., -1:] + 1e-8)  # B, S, P, 2
    x, y = xy.chunk(2, dim=-1)  # B, S, P, 1
    xy = torch.cat([x / Ws * 2 - 1, y / Hs * 2 - 1], dim=-1)  # B, S, P, 2

    # Actual sampling of the image features (along with rgb colors)
    src_feat_rgb = F.grid_sample(src_feat_rgb.view(-1, C, Hs, Ws), xy.view(-1, 1, P, 2), padding_mode=padding_mode).view(B, S, C, P).permute(0, 1, 3, 2)  # BS, C, 1, P -> B, S, C, P -> B, S, P, C
    return src_feat_rgb

    if sample_msk:
        pixel_uv_range = torch.cat([x, y], dim=-1) / src_size[..., None, :] * 2 - 1  # B, S, P, 2 to sample the msk
        should_count_camera = ((pixel_uv_range > -1.0) & (pixel_uv_range < 1.0)).all(dim=-1)  # B, S, P
        should_count_camera = should_count_camera & (src_feat_rgb[..., -1] > 0.5)  # B, S, P # FIXME: best to pass in actual mask for this
        return src_feat_rgb[..., :-1], should_count_camera
    else:
        return src_feat_rgb


@torch.jit.script
def sample_feature_image(xyz: torch.Tensor,
                         src_feat_rgb: torch.Tensor,
                         tar_ext: torch.Tensor,
                         src_exts: torch.Tensor,
                         src_scale: torch.Tensor,
                         src_ixts: torch.Tensor,
                         padding_mode: str = 'border',
                         ):
    # xyz: (B, P, 3)
    # src_feat_rgb: (B, S, C, Hs, Ws)
    B, S, C, Hs, Ws = src_feat_rgb.shape
    B, P, _ = xyz.shape

    src_ixts = src_ixts.clone()  # (B, S, 3, 3)
    src_ixts[..., :2, :] *= src_scale  # (B, S, 3, 3)

    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., -1:])], dim=-1)  # (B, P, 4), homogeneous coordinates
    xyz1 = (xyz1[..., None, :, :] @ src_exts.mT)  # (B, S, P, 4)
    xyzs = xyz1[..., :3] @ src_ixts.mT  # (B, S, P, 3) @ (B, S, 3, 3) -> (B, S, P, 3)
    xy = xyzs[..., :-1] / (xyzs[..., -1:] + 1e-8)  # (B, S, P, 2)
    x, y = xy.chunk(2, dim=-1)  # (B, S, P, 1), (B, S, P, 1)
    xy = torch.cat([x / Ws * 2 - 1, y / Hs * 2 - 1], dim=-1)  # (B, S, P, 2)

    # Actual sampling of the image features (along with rgb colors), (BS, C, 1, P) -> (B, S, C, P) -> (B, S, P, C)
    src_feat_rgb = F.grid_sample(src_feat_rgb.view(-1, C, Hs, Ws), xy.view(-1, 1, P, 2), padding_mode=padding_mode).view(B, S, C, P).permute(0, 1, 3, 2)

    # Computing directional features
    # ENeRF firstly computes direction features and then aggregates them
    # For xyz, use aggregated feature
    # For dir, cat with directional feature
    tar_cam = affine_inverse(tar_ext)[..., :3, -1:].mT  # (B, 1, 3)
    src_cams = affine_inverse(src_exts)[..., :3, -1:].mT  # (B, S, 1, 3)
    tar_ray = normalize(xyz - tar_cam)  # (B, P, 3)
    src_rays = normalize(xyz[:, None] - src_cams)  # (B, S, P, 3)
    ray_diff_dir = normalize(tar_ray[:, None] - src_rays)  # (B, S, P, 3)
    ray_diff_dot = (tar_ray[:, None] * src_rays).sum(dim=-1, keepdim=True)  # (B, S, P, 1)

    src_feat_rgb_dir = torch.cat([src_feat_rgb, ray_diff_dir, ray_diff_dot], dim=-1)
    return src_feat_rgb_dir  # (B, S, P, C)


@torch.jit.script
def sample_feature_volume(s_vals: torch.Tensor, feat_vol: torch.Tensor, ren_scale: torch.Tensor, tar_scale: torch.Tensor,
                          x: int = 0, y: int = 0, w: int = -1, h: int = -1):  # scale are in cpu to avoid sync
    # s_vals: B, P, N
    # feat_vol: B, C, D, Ht, Wt
    B, P, N = s_vals.shape
    B, C, D, Ht, Wt = feat_vol.shape
    WrHr = torch.as_tensor([Wt, Ht])[..., None] / tar_scale * ren_scale
    Wr, Hr = WrHr[0], WrHr[1]
    Wr, Hr = int(Wr.item() + 0.5), int(Hr.item() + 0.5)  # for cpu tensor, no sync, use rounding

    # Use whole image resolution if not specified
    if w < 0: w, h = Wr, Hr
    else: x, y, w, h = int(x * ren_scale[0].item() + 0.5), int(y * ren_scale[1].item() + 0.5), int(w * ren_scale[0].item() + 0.5), int(h * ren_scale[1].item() + 0.5)

    # !: CHUNK
    # Feature volume is w.r.t the target view size
    uv = create_meshgrid(Hr, Wr, device=s_vals.device, dtype=s_vals.dtype, ndc=True).flip(-1)[y:y + h, x:x + w]  # xy
    d = s_vals * 2 - 1  # normalize to -1, 1, B, H, W, N
    uv = uv[None, ..., None, :].expand(B, -1, -1, N, -1)
    d = d.view(B, h, w, N)[..., None]  # B, H, W, N, 1
    uvd = torch.cat([uv, d], dim=-1)  # B, H, W, N, 3

    # B, C, D, H, W
    # B, 3, D, H, W
    vox_feat = F.grid_sample(feat_vol, uvd, padding_mode='border')  # B, C, H, W, N
    vox_feat = vox_feat.permute(0, 2, 3, 4, 1)  # B, H, W, H, C
    return vox_feat.view(B, P * N, -1)  # B, P, C


@torch.jit.script
def depth_regression(depth_prob: torch.Tensor, depth_values: torch.Tensor, volume_render_depth: bool = False, use_dist: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # depth_prob: B, D, H, W
    # depth_values: B, D, H, W

    if volume_render_depth:
        B, D, H, W = depth_prob.shape
        raws = depth_prob.permute(0, 2, 3, 1).reshape(B, H * W, D)  # B, H, W, D -> B, HW, D
        z_vals = depth_values.permute(0, 2, 3, 1).reshape(B, H * W, D)  # B, H, W, D -> B, HW, D
        if use_dist:
            dists = compute_dist(z_vals)  # B, HW, D
            occ = 1. - torch.exp(-raws * dists)  # B, HW, D
        else:
            occ = 1. - torch.exp(-raws)  # B, HW, D
        weights = render_weights(occ)  # B, HW, D
        acc_map = torch.sum(weights, -1, keepdim=True)  # (B, HW, 1)
        depth = weighted_percentile(torch.cat([z_vals, z_vals.max(dim=-1, keepdim=True)[0]], dim=-1),
                                    torch.cat([weights, 1 - acc_map], dim=-1), [0.45, 0.5, 0.55])  # B, HW, 3
        std = (depth[..., -1:] - depth[..., :1]) / 2
        depth = depth[..., 1:2]
        depth = depth.view(B, H, W)
        std = std.view(B, H, W)
    else:
        prob_volume = depth_prob.softmax(-3)  # B, D, H, W
        depth = (prob_volume * depth_values).sum(-3)  # B, H, W
        std = (prob_volume * (depth_values - depth.unsqueeze(-3))**2).sum(-3).clip(1e-10).sqrt()  # B, H, W

    return depth, std


@torch.jit.script
def weight_regression(depth_prob: torch.Tensor, depth_values: torch.Tensor = None, use_dist: bool = False):
    B, D, H, W = depth_prob.shape
    raws = depth_prob.permute(0, 2, 3, 1).view(B, H * W, D)  # B, H, W, D -> B, HW, D
    if use_dist:
        z_vals = depth_values.permute(0, 2, 3, 1).view(B, H * W, D)  # B, H, W, D -> B, HW, D
        dists = compute_dist(z_vals)  # B, HW, D
        occ = 1. - torch.exp(-raws * dists)  # B, HW, D
    else:
        occ = 1. - torch.exp(-raws)  # B, HW, D
    weights = render_weights(occ)  # B, HW, D
    return weights.view(B, H, W, D).permute(0, 3, 1, 2)


@torch.jit.script
def build_cost_vol(depth: torch.Tensor, feat: torch.Tensor, mats: torch.Tensor, agg_method: str = 'var', correct_pix: bool = True):
    # depth: B, D, H, W, sampled feature volume depth
    # feat: B, S, C, H, W, image features
    # homo: B, S, 3, 4, projection matrices

    # Prepare shapes
    sh = feat.shape
    depth = depth[:, None].expand(-1, feat.shape[1], -1, -1, -1)  # no memory allcoation
    depth = depth.reshape((-1, ) + depth.shape[-3:])  # BS, D, H, W, will allocate memory
    feat = feat.view((-1,) + feat.shape[-3:])  # BS, C, H, W
    mats = mats.view(-1, 3, 4)  # BS, 3, 4

    BS, D, Ht, Wt = depth.shape
    BS, C, Hs, Ws = feat.shape

    R = mats[..., :3]  # (BS, 3, 3)
    T = mats[..., 3:]  # (BS, 3, 1)

    # Create grid from the ref frame
    ref_grid = create_meshgrid(Ht, Wt, device=depth.device, correct_pix=correct_pix, dtype=depth.dtype).flip(-1)  # (H, W, 2), in ij ordering
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[..., -1:])), -1)  # H, W, 3
    ref_grid = ref_grid[None].expand(BS, -1, -1, -1).permute(0, 3, 1, 2)  # BS, 3, H, W

    # Warp grid to source views
    src_grid = (R @ ref_grid.reshape(BS, 3, -1))  # apply rotation
    src_grid = src_grid[..., None, :].expand(-1, -1, D, -1).reshape(BS, 3, D * Ht * Wt)  # allocate memory
    src_grid = src_grid + T / depth.reshape(BS, 1, -1)  # BS, 3, DHW
    src_grid = src_grid[..., :-1, :] / src_grid[..., -1:, :].clip(1e-6)  # divide by depth (BS, 2, D*H*W)
    if correct_pix: src_grid = torch.cat([src_grid[..., :1, :] / Ws * 2 - 1, src_grid[..., 1:, :] / Hs * 2 - 1], dim=-2)  # x, y
    else: src_grid = torch.cat([src_grid[..., :1, :] / (Ws - 1) * 2 - 1, src_grid[..., 1:, :] / (Hs - 1) * 2 - 1], dim=-2)  # x, y
    src_grid = src_grid.permute(0, 2, 1)  # (B, D*H*W, 2)
    src_grid = src_grid.view(BS, D, Ht * Wt, 2)  # curious type conversion

    # Actual grid sample
    cost_volume = F.grid_sample(feat, src_grid, padding_mode='border', align_corners=not correct_pix)  # (B, C, D, H*W)

    # Aggregate feature volume
    B, S = sh[:2]  # batch and all source images
    cost_volume = cost_volume.view(B, S, C, D, Ht, Wt)
    if agg_method == 'var': cost_volume = cost_volume.var(dim=1)  # store variance as feature
    elif agg_method == 'sum': cost_volume = cost_volume.sum(dim=1)  # store sum as feature
    elif agg_method == 'mean': cost_volume = cost_volume.mean(dim=1)  # store mean as feature
    elif agg_method == 'median': cost_volume = cost_volume.median(dim=1)[0]  # store median as feature
    elif agg_method == 'noop': pass  # return features directly
    # elif isinstance(agg_method, int): cost_volume = cost_volume[:, agg_method]  # particular index
    else: raise NotImplementedError

    return cost_volume


@torch.jit.script
def build_cost_vol_layer(depth: torch.Tensor, feat: torch.Tensor, mats: torch.Tensor, xywh: List[int], agg_method: str = 'var', correct_pix: bool = True):
    """Cost volume construction, firs step of depth volume construction
    Args:
        depth: torch.Tensor, (B, D, Ht, Wt), sampled feature volume depth, maybe the uniform first layer or guided second layer
        feat: torch.Tensor, (B, S, C, Hs, Ws), source image features
        homo: torch.Tensor, (B, S, 3, 4), target to source projection matrices
        xywh: torch.Tensor, (B, 4), the projected 2d foreground object bounding box
    Returns:
        cost_volume: torch.Tensor, (B, C, D, Hb, Wb), cost volume corresponding to the foreground object bounding box
    """
    assert not (depth.shape[0] > 1), "layered enerf do not support batch size > 1, please set batch size = 1"

    # Prepare shapes
    sh = feat.shape
    depth = depth[:, None].expand(-1, feat.shape[1], -1, -1, -1)  # no memory allcoation
    depth = depth.reshape((-1, ) + depth.shape[-3:])  # BS, D, H, W, will allocate memory
    feat = feat.view((-1,) + feat.shape[-3:])  # BS, C, H, W
    mats = mats.view(-1, 3, 4)  # (B*S, 3, 4)

    x, y, w, h = xywh
    BS, D, Ht, Wt = depth.shape  # Ht, Wt is only used for the whole meshgrid generation
    BS, C, Hs, Ws = feat.shape

    R = mats[..., :3]  # (B*S, 3, 3)
    T = mats[..., 3:]  # (B*S, 3, 1)

    # Create grid from the ref frame
    ref_grid = create_meshgrid(Ht, Wt, device=depth.device, correct_pix=correct_pix, dtype=depth.dtype).flip(-1)[y:y + h, x:x + w, :]  # (Hb, Wb, 2), in xy ordering, # !: batch = 1
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[..., -1:])), -1)  # (Hb, Wb, 3)
    ref_grid = ref_grid[None].expand(BS, -1, -1, -1).permute(0, 3, 1, 2)  # (B*S, 3, Hb, Wb)

    # Warp grid to source views
    src_grid = (R @ ref_grid.reshape(BS, 3, -1))  # apply rotation, (B*S, 3, Hb*Wb)
    src_grid = src_grid[..., None, :].expand(-1, -1, D, -1).reshape(BS, 3, D * h * w)  # allocate memory, (B*S, 3, D*Hb*Wb)
    src_grid = src_grid + T / depth[..., y:y + h, x:x + w].reshape(BS, 1, -1)  # (B*S, 3, D*Hb*Wb)
    src_grid = src_grid[..., :-1, :] / src_grid[..., -1:, :].clip(1e-6)  # divide by depth, (BS, 2, D*Hb*Wb)
    if correct_pix: src_grid = torch.cat([src_grid[..., :1, :] / Ws * 2 - 1, src_grid[..., 1:, :] / Hs * 2 - 1], dim=-2)  # x, y
    else: src_grid = torch.cat([src_grid[..., :1, :] / (Ws - 1) * 2 - 1, src_grid[..., 1:, :] / (Hs - 1) * 2 - 1], dim=-2)  # x, y
    src_grid = src_grid.permute(0, 2, 1)  # (B*S, D*Hb*Wb, 2)
    src_grid = src_grid.view(BS, D, h * w, 2)  # (B*S, D, Hb*Wb, 2)

    # Actual grid sample
    cost_volume = F.grid_sample(feat, src_grid, padding_mode='border', align_corners=not correct_pix)  # (B, C, D, H*W)

    # Aggregate feature volume
    B, S = sh[:2]  # batch and all source images
    cost_volume = cost_volume.view(B, S, C, D, h, w)
    if agg_method == 'var': cost_volume = cost_volume.var(dim=1)  # store variance as feature
    elif agg_method == 'sum': cost_volume = cost_volume.sum(dim=1)  # store sum as feature
    elif agg_method == 'mean': cost_volume = cost_volume.mean(dim=1)  # store mean as feature
    elif agg_method == 'median': cost_volume = cost_volume.median(dim=1)[0]  # store median as feature
    elif isinstance(agg_method, int): cost_volume = cost_volume[:, agg_method]  # particular index
    else: raise NotImplementedError

    return cost_volume


@torch.jit.script
def get_proj_mats(src_scale: torch.Tensor, tar_scale: torch.Tensor,
                  tar_ext: torch.Tensor, tar_ixt: torch.Tensor,
                  src_exts: torch.Tensor, src_ixts: torch.Tensor,
                  ):
    # It's possible for H W to have difference actual scale
    # The configurable scale is only a guidance, since convolutions asks for 4x
    src_ixts = src_ixts.clone()  # B, S, 3, 3
    src_ixts[..., :2, :] *= src_scale
    # B, S, 3, 3 @ B, S, 3, 4 = B, S, 3, 4
    src_projs = src_ixts @ src_exts[..., :3, :]

    tar_ixt = tar_ixt.clone()  # B, 3, 3
    tar_ixt[..., :2, :] *= tar_scale
    tar_projs = tar_ixt @ tar_ext[..., :3, :]  # B, 3, 4
    tar_ones = torch.zeros_like(tar_ext[..., -1:, :])  # B, 1, 4
    tar_ones[..., -1] = 1  # B, 1, 4
    tar_projs = torch.cat((tar_projs, tar_ones), dim=-2)  # B, 4, 4
    tar_projs_inv = torch.inverse(tar_projs.float()).to(tar_projs.dtype)  # B, 4, 4

    # B, S, 3, 4 @ B, 1, 4, 4 = B, S, 3, 4
    proj_mats = src_projs @ tar_projs_inv[..., None, :, :]
    return proj_mats


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_actvn=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_actvn(out_channels)
        self.relu = nn.ReLU(inplace=True)  # might pose problem for pass through optimization, albeit faster

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_actvn=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_actvn(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


@REGRESSORS.register_module()
class FeatureNet(nn.Module):
    def __init__(self, norm_actvn=nn.BatchNorm2d, test_using_train: bool = True):
        super(FeatureNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn

        self.conv0 = nn.Sequential(
            ConvBnReLU(3, 8, 3, 1, 1, norm_actvn=norm_actvn),
            ConvBnReLU(8, 8, 3, 1, 1, norm_actvn=norm_actvn))
        self.conv1 = nn.Sequential(
            ConvBnReLU(8, 16, 5, 2, 2, norm_actvn=norm_actvn),
            ConvBnReLU(16, 16, 3, 1, 1, norm_actvn=norm_actvn))
        self.conv2 = nn.Sequential(
            ConvBnReLU(16, 32, 5, 2, 2, norm_actvn=norm_actvn),
            ConvBnReLU(32, 32, 3, 1, 1, norm_actvn=norm_actvn))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

        self.out_dims = [32, 16, 8]  # output dimensionality
        self.scales = [0.25, 0.5, 1.0]
        self.size_pad = 4  # input size should be divisible by 4
        self.test_using_train = test_using_train

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y

    def forward(self, x: torch.Tensor):
        # x: (B, S, C, H, W) or (B, C, H, W) or (C, H, W)
        # Remember input shapes
        sh = x.shape
        x = x.view(-1, *sh[-3:])  # (B, C, H, W)

        # NOTE: We assume normalized -1, 1 rgb input for feature_net

        # Actual conv net
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        feat2 = self.toplayer(conv2)
        feat1 = self._upsample_add(feat2, self.lat1(conv1))
        feat0 = self._upsample_add(feat1, self.lat0(conv0))
        feat1 = self.smooth1(feat1)
        feat0 = self.smooth0(feat0)

        # Restore original shapes
        feat2 = feat2.view(sh[:-3] + feat2.shape[-3:])
        feat1 = feat1.view(sh[:-3] + feat1.shape[-3:])
        feat0 = feat0.view(sh[:-3] + feat0.shape[-3:])
        return feat2, feat1, feat0  # level0, level1, level2

    def train(self, mode: bool):
        if not mode and self.test_using_train: return
        super().train(mode)


@REGRESSORS.register_module()
class CostRegNet(nn.Module):
    # TODO: compare the results of nn.BatchNorm3d and nn.InstanceNorm3d
    def __init__(self, in_channels, norm_actvn=nn.BatchNorm3d, dpt_actvn=nn.Identity, use_vox_feat=True):
        super(CostRegNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn
        self.dpt_actvn = get_function(dpt_actvn)

        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_actvn=norm_actvn)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_actvn=norm_actvn)
        self.conv2 = ConvBnReLU3D(16, 16, norm_actvn=norm_actvn)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_actvn=norm_actvn)
        self.conv4 = ConvBnReLU3D(32, 32, norm_actvn=norm_actvn)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_actvn=norm_actvn)
        self.conv6 = ConvBnReLU3D(64, 64, norm_actvn=norm_actvn)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(8))
        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))

        self.use_vox_feat = use_vox_feat
        if use_vox_feat:
            self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))

        self.size_pad = 8  # input size should be divisible by 4
        self.out_dim = 8

    def forward(self, x: torch.Tensor):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        depth = self.depth_conv(x)
        depth = self.dpt_actvn(depth.squeeze(1))  # softplus might change dtype

        if self.use_vox_feat:
            feat = self.feat_conv(x)
            return feat, depth
        else:
            return depth


@REGRESSORS.register_module()
class MinCostRegNet(nn.Module):
    def __init__(self, in_channels, norm_actvn=nn.BatchNorm3d, dpt_actvn=nn.Identity):
        super(MinCostRegNet, self).__init__()
        norm_actvn = getattr(nn, norm_actvn) if isinstance(norm_actvn, str) else norm_actvn
        self.dpt_actvn = get_function(dpt_actvn)

        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_actvn=norm_actvn)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_actvn=norm_actvn)
        self.conv2 = ConvBnReLU3D(16, 16, norm_actvn=norm_actvn)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_actvn=norm_actvn)
        self.conv4 = ConvBnReLU3D(32, 32, norm_actvn=norm_actvn)

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_actvn(8))

        self.depth_conv = nn.Sequential(nn.Conv3d(8, 1, 3, padding=1, bias=False))
        self.feat_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=1, bias=False))
        self.out_dim = 8
        self.size_pad = 4  # input should be divisible by 4

    def forward(self, x, use_vox_feat=True):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        depth = self.depth_conv(x)
        depth = self.dpt_actvn(depth.squeeze(1))

        if not use_vox_feat: feat = None
        else: feat = self.feat_conv(x)
        return feat, depth


# ? This could be refactored

class NeRF(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16 + 3):
        """
        """
        super(NeRF, self).__init__()
        self.hid_n = hid_n
        self.agg = FeatureAgg(feat_ch)
        self.lr0 = nn.Sequential(nn.Linear(16, hid_n),
                                 nn.ReLU())
        self.lrs = nn.ModuleList([
            nn.Sequential(nn.Linear(hid_n, hid_n), nn.ReLU()) for i in range(0)
        ])
        self.sigma = nn.Sequential(nn.Linear(hid_n, 1), nn.Softplus())
        self.color = nn.Sequential(
            nn.Linear(64 + 16 + feat_ch + 4, hid_n),
            nn.ReLU(),
            nn.Linear(hid_n, 1),
            nn.ReLU())
        self.lr0.apply(weights_init)
        self.lrs.apply(weights_init)
        self.sigma.apply(weights_init)
        self.color.apply(weights_init)

    def forward(self, vox_feat, img_feat_rgb_dir):
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        img_feat = self.agg(img_feat_rgb_dir)
        S = img_feat_rgb_dir.shape[2]

        vox_img_feat = torch.cat((vox_feat, img_feat), dim=-1)

        x = self.lr0(vox_img_feat)
        for i in range(len(self.lrs)):
            x = self.lrs[i](x)
        sigma = self.sigma(x)
        x = torch.cat((x, vox_img_feat), dim=-1)
        x = x.view(B, -1, 1, x.shape[-1]).repeat(1, 1, S, 1)
        x = torch.cat((x, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x), dim=-2)
        color = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        return torch.cat([color, sigma], dim=-1)


@REGRESSORS.register_module()
class FeatureAgg(nn.Module):
    def __init__(self, feat_ch, viewdir_agg=True):
        """
        """
        super(FeatureAgg, self).__init__()
        self.feat_ch = feat_ch

        # Layered ENeRF ignores viewdir during vanilla xyz embedding
        self.viewdir_agg = viewdir_agg
        if self.viewdir_agg:
            self.view_fc = nn.Sequential(
                nn.Linear(4, feat_ch),
                nn.ReLU(),
            )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
            nn.Linear(feat_ch * 3, 32),
            nn.ReLU(),
        )

        self.agg_w_fc = nn.Sequential(
            nn.Linear(32, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

        self.out_dim = 16

    def forward(self, img_feat_rgb_dir: torch.Tensor):
        # Prepare shapes
        img_feat_rgb_dir = img_feat_rgb_dir.permute(0, 2, 1, 3)  # B, S, P, C -> B, P, S, C

        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]

        if self.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., self.feat_ch:])
            img_feat_rgb = img_feat_rgb_dir[..., :self.feat_ch] + view_feat
        else:
            img_feat_rgb = img_feat_rgb_dir[..., :self.feat_ch]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).expand(-1, -1, S, -1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).expand(-1, -1, S, -1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)  # B, P, C


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


@REGRESSORS.register_module()
class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 coarse_out_ch=32,
                 fine_out_ch=32,
                 norm_actvn=nn.InstanceNorm2d,
                 coarse_only=False,
                 **kwargs
                 ):

        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.coarse_only = coarse_only
        if self.coarse_only:
            fine_out_ch = 0
        self.coarse_out_ch = coarse_out_ch
        self.fine_out_ch = fine_out_ch
        out_ch = coarse_out_ch + fine_out_ch

        # original
        layers = [3, 4, 6, 3]
        self._norm_layer = norm_actvn
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='reflect')
        self.bn1 = norm_actvn(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

        self.out_dims = [32, 32]  # output dimensionality
        self.scales = [0.125, 0.25]
        self.size_pad = 1  # input size should be divisible by 4

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)

        if self.coarse_only:
            x_coarse = x_out
            x_fine = None
            return [x_coarse]
        else:
            x_fine = x_out[:, :self.fine_out_ch, :]
            x_coarse = x_out[:, -self.coarse_out_ch:, :]
            return [x_fine, x_coarse]
