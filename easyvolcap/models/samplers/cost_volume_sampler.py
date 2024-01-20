import torch
import numpy as np
from typing import List

from torch import nn
from functools import lru_cache
from torch.nn import functional as F
from torchvision.io import decode_jpeg

from easyvolcap.engine import cfg, args
from easyvolcap.engine import SAMPLERS, REGRESSORS
from easyvolcap.models.networks.embedders.image_based_embedder import ImageBasedEmbedder
from easyvolcap.models.networks.multilevel_network import MultilevelNetwork  # use unified network implementation to aggregate similar functionalities
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork
from easyvolcap.models.samplers.uniform_sampler import UniformSampler

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer  # global timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import to_x
from easyvolcap.utils.ibr_utils import get_src_inps, get_src_feats, prepare_caches, compute_src_inps, compute_src_feats
from easyvolcap.utils.nerf_utils import linear_sampling, ray2xyz, volume_rendering
from easyvolcap.utils.prop_utils import s_vals_to_z_vals, z_vals_to_s_vals
from easyvolcap.utils.image_utils import interpolate_image, pad_image
from easyvolcap.utils.enerf_utils import MinCostRegNet, CostRegNet, FeatureNet, get_proj_mats, build_cost_vol, depth_regression, render_debug_cost_volume

# ? How do we share 2D convolutions between multiple levels?
# We use a single level network for this, allow more customization since cas_enerf isn't multiple NeRFs stacked together -> instead they are highly intertwined
# We store coarse level color in regressor, and fine level color as direct output
# The dir embedder could reuse output of image_based_regressor
# Maybe we should add switch about what kind of output to expect
# Add more checking in the batch for already present features

# Feature list:
# 1. [*]src_feat, feature for all source images, List[torch.Tensor], B, H, W, S, C
# 2. [*]feat_volume, feature volume (grid), List[torch.Tensor], B, H, W, D, C
# 3. [*]img_feat, point image feature, sampled, another method of embedder
# 4. [*]vox_feat, point feature volume

# NOTE: src_feat level are not 1 to 1 correspondent to feat_volume level, there should exist a user controllable mapping


@SAMPLERS.register_module()
class CostVolumeSampler(UniformSampler):
    def __init__(self,
                 network: MultilevelNetwork,

                 feat_cfg: dotdict = dotdict(type=FeatureNet.__name__),
                 cost_reg_cfgs: List[dotdict] = [
                     dotdict(type=MinCostRegNet.__name__),  # two levels of cost volume
                     dotdict(type=CostRegNet.__name__),  # two levels of cost volume
                 ],

                 n_samples=[8, 2],  # number of actual color samples
                 n_planes=[64, 8],  # number of depth values?
                 uni_disps=[True, False],  # sample in uniform disparity
                 render_if=[True, True],  # how do we supervise this?
                 vol_scales=[0.25, 1.0],  # used for cost volume
                 ren_scales=[0.25, 1.0],  # used for rendering
                 ibr_level_map=[0, 2],  # use other level for ibr

                 bg_brightness: float = -1.0,
                 dist_default: float = 1.0,
                 volume_render_depth: bool = False,  # ENeRF: False, ENeRF++: True

                 cache_size: int = 10 if args.type != 'train' else 0,  # (512 * 512 * 3 * 8 + 256 * 256 * 3 * 16 + 128 * 128 * 3 * 32) * 4 / 2 ** 20 = 42.0 MB -> all cached -> 26 GB of VRAM
                 dtype: str = torch.float,

                 **kwargs,
                 ):
        super().__init__(network=network, **kwargs)  # will remember self.network
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.feat_reg: FeatureNet = REGRESSORS.build(feat_cfg).to(self.dtype)  # for 2d image feature extraction
        self.cost_regs = nn.ModuleList([
            REGRESSORS.build(cost_reg_cfg, in_channels=self.feat_reg.out_dims[i]).to(self.dtype)
            for i, cost_reg_cfg in enumerate(cost_reg_cfgs)
        ])

        self.n_samples = n_samples
        self.n_planes = n_planes
        self.render_if = render_if
        self.vol_scales_guide = vol_scales  # only a guidance for final shape
        self.ren_scales_guide = ren_scales  # only a guidance for final shape
        self.img_scales_guide = self.feat_reg.scales  # only a guidance for final shape, 0.25, 0.5
        self.vol_pads = [cost_reg.size_pad for cost_reg in self.cost_regs]
        self.img_pad = self.feat_reg.size_pad
        self.ibr_level_map = ibr_level_map

        self.volume_render_depth = volume_render_depth
        self.bg_brightness = bg_brightness
        self.dist_default = dist_default

        # Prepare for depth_curve_fn
        def identity(x): return x
        def reciprocal(x): return (1 / (x + 1e-10))
        self.gs = [reciprocal if uni_disp else identity for uni_disp in uni_disps]
        self.igs = [reciprocal if uni_disp else identity for uni_disp in uni_disps]

        # Levels should match, there might more elegant ways of implementing this
        assert len(n_samples) == len(n_planes)
        assert len(n_samples) == len(uni_disps)
        assert len(n_samples) == len(render_if)
        assert len(n_samples) == len(vol_scales)
        assert len(n_samples) == len(ren_scales)
        assert len(n_samples) == len(ibr_level_map)
        # assert len(n_samples) == len(cost_reg_cfgs)
        assert ren_scales[-1] == 1.0  # last level should be of full resolution

        # Prepare for lru_cache of gpu memory
        self.cache_size = cache_size
        prepare_caches(cache_size)  # define cache size

    def sample_depth(self,
                     ray_o: torch.Tensor, ray_d: torch.Tensor,
                     near: torch.Tensor, far: torch.Tensor, t: torch.Tensor,
                     batch: dotdict):

        timer.record('network forwarding')

        # Prepare inputs and features
        src_inps = compute_src_inps(batch) if self.training else get_src_inps(batch)  # will inplace upate feature
        src_inps = src_inps.to(self.dtype)
        timer.record('get src inps')

        # Values to be reused
        output = dotdict()

        # For supervision of coarse network
        output.Hrs_prop = []
        output.Wrs_prop = []
        output.rgb_maps_prop = []
        output.bg_colors_prop = []

        # Prepare for shape (ENeRF requires image shaped things)
        B, _, _ = ray_o.shape
        Ho, Wo = batch.meta.H[0].item(), batch.meta.W[0].item()  # target view size (without any scaling)
        Hr, Wr = Ho, Wo  # -1 level rendering size

        # Preparing source scaling (for painless up convolution and skip connections)
        Hc, Wc = src_inps.shape[-2:]  # cropped image size
        Hp, Wp = int(np.ceil(Hc / self.img_pad)) * self.img_pad, int(np.ceil(Wc / self.img_pad)) * self.img_pad  # Input and output should be same in size
        src_inps = pad_image(src_inps, size=(Hp, Wp))  # B, S, 3, H, W
        timer.record('fill src imgs')

        # Forward feature extraction
        # `src_feats` is a list of features of shape (B, S, C, H, W) -> (B, S, 32*(2**(-i)), H//4*(2**i), W//4*(2**i))
        src_feats = compute_src_feats(src_inps, self.feat_reg, batch) if self.training else get_src_feats(src_inps, self.feat_reg, batch)
        timer.record('get src feats')

        # Preparing for camera parameters
        tar_ext, tar_ixt, src_exts, src_ixts = batch.tar_ext, batch.tar_ixt, batch.src_exts, batch.src_ixts
        tar_ext, tar_ixt, src_exts, src_ixts = to_x([tar_ext, tar_ixt, src_exts, src_ixts], self.dtype)

        for level in range(len(self.n_samples)):  # iterate through all levels
            # Preparing source scaling (for painless up convolution and skip connections)
            Ht, Wt = int(Ho * self.vol_scales_guide[level]), int(Wo * self.vol_scales_guide[level])  # applied on original image
            Ht, Wt = int(np.ceil(Ht / self.vol_pads[level])) * self.vol_pads[level], int(np.ceil(Wt / self.vol_pads[level])) * self.vol_pads[level]
            tar_scale = ray_o.new_empty(2, 1, dtype=self.dtype)  # 2, 1
            tar_scale[0] = Wt / Wo
            tar_scale[1] = Ht / Ho
            tar_scale_cpu = torch.as_tensor([Wt / Wo, Ht / Ho])[..., None]  # 2, 1

            # Preparing target image scaling
            Hs, Ws = int(Hp * self.img_scales_guide[level]), int(Wp * self.img_scales_guide[level])  # applied on feature image
            src_scale = ray_o.new_empty(2, 1, dtype=self.dtype)  # 2, 1
            src_scale[0] = Ws / Wp
            src_scale[1] = Hs / Hp
            src_scale_cpu = torch.as_tensor([Ws / Wp, Hs / Hp])[..., None]  # 2, 1

            # Let's have a look at all three scaling factors to better understand how it operates:
            # 1. [*]vol_scales_guide = [0.125, 0.5]: resolution of the cost volume, namely the scaling factor of the target image since cost volume is built upon the target camera frustrum.
            # 2. [*]img_scales_guide = [ 0.25, 0.5]: resolution of the feature maps, namely the resolution of the first and second level of the feature pyramid(`src_feats`), the cost volume is built by first projecting the `vol_scales_guide = [0.125, 0.5]` scaled 2d target pixels into the corresponding 2d source feature pyramid of resolution `[0.25, 0.5]`, and then interpolating the corresponding feature pyramid as its local feature, finally compute the feature average and variance alongside S dim.
            # 3. [*]ren_scales_guide = [ 0.25, 1.0]: resolution of the rendering image results for each level
            # So, now it is much more clear why ENeRF needs three scaling factors, especially the previous two, `vol_scales_guide` and `vol_scales_guide`.

            # Depth plane sampling: 0, 1 sampling -> near far plane sampling
            if level == 0:
                # Volume built on camera frustrum
                near_plane = near.min(dim=-2)[0][..., None, None, :]  # B, P, 1 -> B, 1, 1, 1
                far_plane = far.max(dim=-2)[0][..., None, None, :]  # B, P, 1 -> B, 1, 1, 1
            else:
                # Irregular volume built around the surface
                near_plane = interpolate_image(near, size=(Ht, Wt))  # ? SAMPLE
                far_plane = interpolate_image(far, size=(Ht, Wt))  # ? SAMPLE
            s_vals = linear_sampling(B, Ht, Wt, self.n_planes[level], device=near.device, dtype=near.dtype, perturb=False).permute(0, 3, 1, 2)  # B, D, Ht, Wt == B, D, Ho*vol_scale_l, Wo*vol_scale_l
            z_vals = s_vals_to_z_vals(s_vals, near_plane, far_plane, self.gs[level], self.igs[level])  # B, D, Ht, Wt == B, D, Ho*vol_scale_l, Wo*vol_scale_l
            timer.record('near far planes')

            # Depth volume construction: forward cost volume regressor -> perform depth regression
            proj_mats = get_proj_mats(src_scale, tar_scale, tar_ext, tar_ixt, src_exts, src_ixts)  # B, S, 3, 4, tgt2src relative projection matrix
            timer.record('get proj mats')

            cost_vol = build_cost_vol(z_vals, src_feats[level], proj_mats)  # B, C, D, Ht, Wt == B, C, D, Ho*vol_scale_l, Wo*vol_scale_l
            timer.record('build cost vol')

            feat_vol, depth_prob = self.cost_regs[level](cost_vol)  # B, C, D, Ht, Wt; B, D, Ht, Wt
            timer.record('cost vol reg')

            depth, std = depth_regression(depth_prob, z_vals, self.volume_render_depth)  # B, Ht, Wt; B, Ht, Wt (maybe perform regression in disp space)
            timer.record('depth regression')

            # Sample actual depth for network forward
            near = (depth - std)[:, None].clip(near_plane, far_plane)  # B, 1, Ht, Wt, maybe disp depth range
            far = (depth + std)[:, None].clip(near_plane, far_plane)  # B, 1, Ht, Wt, maybe disp depth range

            if (not self.training or not self.render_if[level]) and level != len(self.n_samples) - 1: continue  # only perform multi-level depth sampling if training and requires render

            # Prepare for target image rendering scale
            Hr, Wr = int(Ho * self.ren_scales_guide[level]), int(Wo * self.ren_scales_guide[level])  # applied on original image
            ren_scale = ray_o.new_empty(2, 1, dtype=self.dtype)  # 2, 1
            ren_scale[0] = Wr / Wo
            ren_scale[1] = Hr / Ho
            ren_scale_cpu = torch.as_tensor([Wr / Wo, Hr / Ho])[..., None]  # 2, 1

            # Preparing source image scaling
            Hs, Ws = int(Hp * self.img_scales_guide[self.ibr_level_map[level]]), int(Wp * self.img_scales_guide[self.ibr_level_map[level]])  # applied on feature image
            src_scale = ray_o.new_empty(2, 1, dtype=self.dtype)  # 2, 1
            src_scale[0] = Ws / Wp
            src_scale[1] = Hs / Hp
            src_scale_cpu = torch.as_tensor([Ws / Wp, Hs / Hp])[..., None]  # 2, 1

            # Sample points for volume rendering
            near = interpolate_image(near, size=(Hr, Wr))  # B, 1, Hr, Wr # TODO: PERF
            far = interpolate_image(far, size=(Hr, Wr))  # B, 1, Hr, Wr # TODO: PERF
            timer.record('near far scaling')

            s_vals = linear_sampling(B, Hr, Wr, self.n_samples[level], device=near.device, dtype=near.dtype, perturb=self.training).permute(0, 3, 1, 2)  # 2 samples
            z_vals = s_vals_to_z_vals(s_vals, near, far, self.gs[level], self.igs[level])
            s_vals = s_vals.permute(0, 2, 3, 1).reshape(B, Hr * Wr, -1)  # B, P, N
            z_vals = z_vals.permute(0, 2, 3, 1).reshape(B, Hr * Wr, -1)  # B, P, N
            timer.record('sampling preparation')

            # Preparing for output (maybe coarse (later), or fine (return))
            # Here, the resolution of the `src_feats[self.ibr_level_map[level]]` is the same as the target rendered image, namely [0.25, 1.0]
            src_feat = src_feats[self.ibr_level_map[level]]  # for rendering image, denisty and blend weight decoding
            src_inps_rgb = interpolate_image(src_inps, size=(Hs, Ws))  # for rendering image, image-based rendering # TODO: PERF
            src_feat_rgb = torch.cat([src_feat, src_inps_rgb], dim=-3)  # B, S, C, Hr, Wr
            timer.record('rgb prepration')

            # For proposal loss (also need rendered weights)
            output.feat_vol = feat_vol  # B, C, D, Ht, Wt
            output.src_feat_rgb = src_feat_rgb  # B, S, C, Hr, Wr
            output.s_vals = s_vals  # B, P, N
            output.z_vals = z_vals  # B, P, N
            output.tar_scale = tar_scale  # vol_scales_guide = [0.125, 0.5]
            output.src_scale = src_scale  # img_scales_guide[ibr_level_map] = ren_scales_guide = [0.25, 1.0]
            output.ren_scale = ren_scale  # ren_scales_guide = [0.25, 1.0]
            output.meta.tar_scale = tar_scale_cpu
            output.meta.src_scale = src_scale_cpu
            output.meta.ren_scale = ren_scale_cpu
            timer.record('output preparation')

            # For the finest level, use model's rendering logic (network -> renderer)
            if level == len(self.n_samples) - 1: break  # the last(finest) level doesn't require any network evals here

            # Perform network forwarding for occ? for rgb?
            # OCC output -> weight -> proposal loss
            # RGB output -> ask volume renderer to render? -> rgb loss?
            ray_o_c = interpolate_image(ray_o.view(B, Ho, Wo, 3).permute(0, 3, 1, 2), size=(Hr, Wr)).permute(0, 2, 3, 1).view(B, -1, 3)
            ray_d_c = interpolate_image(ray_d.view(B, Ho, Wo, 3).permute(0, 3, 1, 2), size=(Hr, Wr)).permute(0, 2, 3, 1).view(B, -1, 3)
            t_c = interpolate_image(t.view(B, Ho, Wo, 1).permute(0, 3, 1, 2), size=(Hr, Wr)).permute(0, 2, 3, 1).view(B, -1, 1)
            xyz_c, dir_c, t_c, dist_c = ray2xyz(ray_o_c, ray_d_c, t_c, z_vals, self.dist_default)
            batch.output.update(output)  # NOTE: temporarily update batch to perform the forward pass

            # Render both occ and rgb for rgb loss
            rgb, occ = self.network.compute_coarse(VolumetricVideoNetwork.compute.__name__, level, xyz_c, dir_c, t_c, dist_c, batch)

            # Nasty shapes
            sh = z_vals.shape  # B, P, S
            occ = occ.view(sh + (-1,))  # B, P, S, 1
            rgb = rgb.view(sh + (-1,))  # B, P, S, 3
            if self.bg_brightness < 0 and self.training: bg_color = torch.rand_like(rgb[..., 0, :])  # remove sample dim (in 0 -> 1 already)
            else: bg_color = torch.full_like(rgb[..., 0, :], self.bg_brightness)  # remove sample dim
            weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_color)  # B, P, S

            # For supervision of coarse network
            output.Hrs_prop.append(Hr)
            output.Wrs_prop.append(Wr)
            output.rgb_maps_prop.append(rgb_map)
            output.bg_colors_prop.append(bg_color)

        batch.output.update(output)  # For performing the actual forward pass
        return z_vals

    def sample(self, ray_o: torch.Tensor, ray_d: torch.Tensor, near: torch.Tensor, far: torch.Tensor, t: torch.Tensor, batch: dotdict):
        z_vals = self.sample_depth(ray_o, ray_d, near, far, t, batch)
        xyz, dir, t, dist = ray2xyz(ray_o, ray_d, t, z_vals, self.dist_default)  # B, P * S, 3 (for easy chunking)
        return xyz, dir, t, dist
