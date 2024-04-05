import torch
import numpy as np
import torch.nn.functional as F
from typing import Union, List, Tuple

from torch import nn
from functools import lru_cache
from torchvision.io import decode_jpeg

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.ray_utils import create_meshgrid
from easyvolcap.utils.math_utils import affine_inverse, normalize
from easyvolcap.utils.image_utils import pad_image, interpolate_image
from easyvolcap.utils.data_utils import export_pts, export_pcd, export_mesh
from easyvolcap.utils.prop_utils import s_vals_to_z_vals, weighted_percentile
from easyvolcap.utils.nerf_utils import linear_sampling, ray2xyz, volume_rendering, raw2alpha, compute_dist, render_weights


# How do we easily wraps all stuff in the lru?
# Initialize the lru requires some change to global variables, better expose APIs
# Input should only be latent_index and view_index, also use globals to manage other inputs
g_batch: dotdict = None
g_feat_regs = dotdict(g_feat_reg=None, g_bg_feat_reg=None)
g_src_inps = dotdict(src_inp=None, bg_src_inp=None)


def g_cached_input(latent_index: int, view_index: int, key: str = 'src_inps'):  # MARK: INPLACE
    global g_src_inps
    g_src_inp = g_src_inps.src_inp if key == 'src_inps' else g_src_inps.bg_src_inp
    if g_src_inp.ndim == 2:  # B, N or B, 3, H, W
        g_src_inp = decode_jpeg(g_src_inp[0].cpu(), device='cuda')[None].float() / 255  # decode -> 3, H, W, uint8(0,255) -> float32(0,1)
    return g_src_inp


def g_cached_feature(latent_index: int, view_index: int, key: str = 'fg'):
    g_feat_reg = g_feat_regs.g_bg_feat_reg if key == 'bg' else g_feat_regs.g_feat_reg
    g_src_inp = g_src_inps.src_inp
    return g_feat_reg(g_src_inp * 2 - 1)  # S, C, H, W


def prepare_caches(maxsize: int = 512):
    global g_cached_feature, g_cached_input
    g_cached_input = lru_cache(maxsize)(g_cached_input)
    g_cached_feature = lru_cache(maxsize)(g_cached_feature)


# NOTE: Everything starting with `get` is a cached function


def get_src_inps(batch: dotdict, key: str = 'src_inps'):  # MARK: INPLACE # !: BATCH
    global g_src_inps
    if key not in batch.meta:
        return compute_src_inps(batch, key)
    src_inps = batch.meta[key]

    inps = []
    t = batch.meta.t_inds[0]  # B, -> ()
    src_inds = batch.meta.src_inds[0]  # B, S -> S,
    for i, v in enumerate(src_inds):
        if key == 'src_inps': g_src_inps.src_inp = src_inps[i]  # jpeg bytes or tensors
        else: g_src_inps.bg_src_inp = src_inps[i]  # jpeg bytes or tensors
        inp = g_cached_input(t.item(), v.item(), key)  # B, 3, H, W
        inps.append(inp)
    batch[key] = inps  # list of B, 3, H, W
    return compute_src_inps(batch, key)


def get_src_feats(src_inps: torch.Tensor, feat_reg: nn.Module, batch: dotdict, key: str = 'fg'):  # !: BATCH
    global g_src_inps, g_feat_regs
    if key == 'fg': g_feat_regs.g_feat_reg = feat_reg
    else: g_feat_regs.g_bg_feat_reg = feat_reg
    src_inps = src_inps[0]  # !: BATCH

    Hc, Wc = src_inps.shape[-2:]  # cropped image size
    Hp, Wp = int(np.ceil(Hc / feat_reg.size_pad)) * feat_reg.size_pad, int(np.ceil(Wc / feat_reg.size_pad)) * feat_reg.size_pad  # Input and output should be same in size
    Hps = [int(Hp * s + 1e-5) for s in feat_reg.scales]
    Wps = [int(Wp * s + 1e-5) for s in feat_reg.scales]

    feats = []
    t = batch.meta.t_inds[0]  # B, -> ()
    src_inds = batch.meta.src_inds[0]  # B, S -> S,
    for i, v in enumerate(src_inds):
        g_src_inps.src_inp = src_inps[i]  # use full image for both foreground and background (layer enerf experience)
        fs = g_cached_feature(t.item(), v.item(), key)
        fs = [pad_image(f, size=(h, w)) for h, w, f in zip(Hps, Wps, fs)]  # resizing to current input
        feats.append(fs)
    feats = [torch.stack([f[i] for f in feats])[None] for i in range(len(feat_reg.scales))]  # HACK: too hacky...

    return feats


# Plain function for loading or computing the src_inps and src_feats
# NOTE: Everything starting with `compute` is a plain function without cache

def compute_src_inps(batch: dotdict, key: str = 'src_inps'):
    # [Storage] Input image might have completely different size from each other from the get go
    # [Dataset] Given differently sized images, performs cropping and scaling, update intrinsics, store as jpeg bytes -> jpeg streams are small
    # [Sampler] Decodes stored images (using torchvision api of nvjpeg), stack them together with zero padding -> images might be large
    # [Sampler] Forwards images to interpolation, pass through feature decoder -> feature might be large

    # The `key` may be `src_inps` or `src_bkgs` for enerf and layer enerf respectively
    if key not in batch and key in batch.meta and isinstance(batch.meta[key], list) and batch.meta[key][0].ndim == 2:
        # Perform decoding
        # The input tensor must be on CPU when decoding with nvjpeg
        # decode_jpeg already outputs 3, H, W images, only accepts 1d uint8
        batch[key] = [torch.cat([decode_jpeg(inp.cpu(), device='cuda') for inp in inps])[None].float() / 255 for inps in batch.meta[key]]  # decode -> 3, H, W, uint8(0,255) -> float32(0,1)

    if isinstance(batch[key], list) and batch[key][0].ndim == 4:  # S: B, C, H, W -> B, S, C, H, W
        # Perform reshaping
        max_h = max([i.shape[-2] for i in batch[key]])
        max_w = max([i.shape[-1] for i in batch[key]])
        batch[key] = torch.stack([pad_image(img, [max_h, max_w]) for img in batch[key]]).permute(1, 0, 2, 3, 4)  # S: B, C, H, W -> B, S, C, H, W

    return batch[key].contiguous()  # always return a contiguous tensor


def compute_src_feats(src_inps: torch.Tensor, feat_reg: nn.Module, batch: dotdict = None):
    # [Sampler] Maybe perform convolution on the input src_inps (if training)
    # [Sampler] Or just load it from the dataset, is preloading feature the best practice?
    sh = src_inps.shape
    src_inps = src_inps.view(-1, *sh[-3:])
    feats = feat_reg(src_inps * 2 - 1)  # always return a tensor
    feats = [f.view(*sh[:-3], *f.shape[-3:]) for f in feats]
    return feats


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
def sample_geometry_feature_image(xyz: torch.Tensor,  # B, P, 3
                                  src_feat_rgb: torch.Tensor,  # B, S, C, H, W
                                  src_exts: torch.Tensor,  # B, S, 3, 4
                                  src_ixts: torch.Tensor,  # B, S, 3, 3
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
