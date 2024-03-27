# Point planes IBR sampler
# Use the point as geometry
# Use K-planes as feature bases
# Use IBR for rendering the final rgb color -> hoping for a sharper result

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.net_utils import make_params, make_buffer

from easyvolcap.engine import cfg
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.engine import SAMPLERS, EMBEDDERS, REGRESSORS

from easyvolcap.models.samplers.gaussiant_sampler import GaussianTSampler
from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor
from easyvolcap.models.samplers.point_planes_sampler import PointPlanesSampler
from easyvolcap.models.networks.embedders.kplanes_embedder import KPlanesEmbedder
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork
from easyvolcap.models.networks.regressors.spherical_harmonics import SphericalHarmonics
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder
from easyvolcap.models.networks.embedders.geometry_image_based_embedder import GeometryImageBasedEmbedder
from easyvolcap.models.networks.regressors.image_based_spherical_harmonics import ImageBasedSphericalHarmonics


@SAMPLERS.register_module()
class R4DVSampler(PointPlanesSampler):
    def __init__(self,
                 network: VolumetricVideoNetwork,  # always as the first argument of sampler
                 use_diffgl: bool = True,

                 ibr_embedder_cfg: dotdict = dotdict(type=GeometryImageBasedEmbedder.__name__),  # easily returns nan
                 ibr_regressor_cfg: dotdict = dotdict(type=ImageBasedSphericalHarmonics.__name__),  # easily returns nan

                 opt_cnn_warmup: int = 1000,  # optimize for 1000 iterations
                 opt_cnn_every: int = 100,  # optimize every 100 iterations after
                 render_gs: bool = False,

                 **kwargs,
                 ):
        kwargs = dotdict(kwargs)
        self.kwargs = kwargs

        call_from_cfg(super().__init__, kwargs, network=network, use_diffgl=use_diffgl)  # later arguments will overwrite former ones
        del self.dir_embedder  # no need for this
        del self.rgb_regressor
        self.ibr_embedder: GeometryImageBasedEmbedder = EMBEDDERS.build(ibr_embedder_cfg)  # forwarding the images
        self.ibr_regressor: ImageBasedSphericalHarmonics = REGRESSORS.build(ibr_regressor_cfg, in_dim=self.xyz_embedder.out_dim + 3, src_dim=self.ibr_embedder.src_dim)
        self.pre_handle = self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.type(self.dtype)

        self.opt_cnn_warmup = opt_cnn_warmup
        self.opt_cnn_every = opt_cnn_every

        self.render_radius = MethodType(GaussianTSampler.render_radius, self)  # override the method
        self.sh_deg = 0  # only input colors
        self.scale_mod = 1.0
        self.render_gs = render_gs

    def render_points(self, xyz: torch.Tensor, rgb: torch.Tensor, rad: torch.Tensor, occ: torch.Tensor, batch: dotdict):
        if self.render_gs:
            sh0 = (rgb[..., None] - 0.5) / 0.28209479177387814
            rgb, acc, dpt = self.render_radius(xyz, sh0, rad, occ, batch)  # B, HW, C
        else:
            rgb, acc, dpt = super().render_points(xyz, rgb, rad, occ, batch)  # almost always use render_cudagl
        return rgb, acc, dpt

    def type(self, dtype: torch.dtype):
        super().type(dtype)
        if hasattr(self, 'pcd_embedder'):
            if self.pcd_embedder.spatial_embedding[0].tcnn_encoding.dtype != dtype:
                prev_pcd_embedder = self.pcd_embedder
                self.pcd_embedder: KPlanesEmbedder = EMBEDDERS.build(self.kwargs.pcd_embedder_cfg, dtype=dtype)  # unchanged and loaded as is
                self.pcd_embedder.load_state_dict(prev_pcd_embedder.state_dict())
                self.pcd_embedder.to(prev_pcd_embedder.bounds.device)
            else:
                self.pcd_embedder.xy.data = self.pcd_embedder.xy.to(torch.long)
                self.pcd_embedder.xz.data = self.pcd_embedder.xz.to(torch.long)
                self.pcd_embedder.yz.data = self.pcd_embedder.yz.to(torch.long)

        if hasattr(self, 'xyz_embedder'):
            if self.xyz_embedder.spatial_embedding[0].tcnn_encoding.dtype != dtype:
                prev_xyz_embedder = self.xyz_embedder
                self.xyz_embedder: KPlanesEmbedder = EMBEDDERS.build(self.kwargs.xyz_embedder_cfg, dtype=dtype)  # unchanged and loaded as is
                self.xyz_embedder.load_state_dict(prev_xyz_embedder.state_dict())
                self.xyz_embedder.to(prev_xyz_embedder.bounds.device)
            else:
                self.xyz_embedder.xy.data = self.xyz_embedder.xy.to(torch.long)
                self.xyz_embedder.xz.data = self.xyz_embedder.xz.to(torch.long)
                self.xyz_embedder.yz.data = self.xyz_embedder.yz.to(torch.long)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # Historical reasons
        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.feat_agg' in key:
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.rgb_mlp.linears' in key:
                state_dict[key.replace(f'{prefix}ibr_regressor.rgb_mlp.linears', f'{prefix}ibr_regressor.rgb_mlp.mlp.linears')] = state_dict[key]
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.sh_mlp.linears' in key:
                state_dict[key.replace(f'{prefix}ibr_regressor.sh_mlp.linears', f'{prefix}ibr_regressor.sh_mlp.mlp.linears')] = state_dict[key]
                del state_dict[key]

    def forward(self, batch: dotdict, return_frags: bool = False):
        timer.record('post processing')
        self.init_points(batch)
        self.update_points(batch)
        pcd, pcd_t = self.sample_pcd_pcd_t(batch)  # B, P, 3, B, P, 1

        # These could be discarded
        pcd_feat = self.pcd_embedder(pcd, pcd_t)  # B, N, C
        resd = self.resd_regressor(pcd_feat)  # B, N, 3
        xyz = pcd + resd  # B, N, 3

        # These could be cached -> or could it be expanded?
        xyz_feat = self.xyz_embedder(xyz, pcd_t)  # same time

        # These could be stored
        rad, occ = self.geo_regressor(xyz_feat)  # B, N, 1
        timer.record('geometry')

        # These could be cached on points
        optimize_cnn = not (batch.meta.iter % self.opt_cnn_every) or (batch.meta.iter <= self.opt_cnn_warmup)
        src_feat = self.ibr_embedder(xyz, batch, optimize_cnn=optimize_cnn)  # MARK: implicit update of batch.output
        dir = normalize(xyz.detach() - (-batch.R.mT @ batch.T).mT)  # B, N, 3
        rgb = self.ibr_regressor(torch.cat([xyz_feat, dir], dim=-1), batch)  # B,  N, 3
        timer.record('appearance')

        if return_frags:
            return pcd, xyz, rgb, rad, occ

        # Perform points rendering
        rgb, acc, dpt = self.render_points(xyz, rgb, rad, occ, batch)  # B, HW, C
        self.store_output(pcd, xyz, rgb, acc, dpt, batch)
        timer.record('rendering')
