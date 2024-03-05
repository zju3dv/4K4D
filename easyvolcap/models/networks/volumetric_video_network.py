# Default optimizable parameter pipeline for volumetric videos
# Since it's volumetric, we define the 6d plenoptic function
# So no funny ray parameterization here (plucker, etc)
# This should be the most general purpose implementation
# parameterization (xyztθφ -> xyztθφ) ->
# xyzt embedder (xyzt -> feat) ->
# deformer (xyztθφ, feat -> xyztθφ) ->
# xyz embedder (xyzt -> feat) ->
# geometry (feat -> occ, feat) ->
# tθφ embedder (tθφ -> feat) ->
# appearance (feat, feat -> rgb)

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import torch
from torch import nn
from easyvolcap.engine import cfg
from easyvolcap.engine import NETWORKS, EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import Visualization
from easyvolcap.utils.net_utils import GradientModule
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.blend_utils import apply_rt
from easyvolcap.utils.nerf_utils import raw2alpha
from easyvolcap.utils.blend_utils import apply_rt

# Only for type annotation and default arguments
# Isn't this a little bit clumsy? Retyping the network name everytime?
from easyvolcap.models.networks.embedders.empty_embedder import EmptyEmbedder
from easyvolcap.models.networks.embedders.spacetime_embedder import SpacetimeEmbedder
from easyvolcap.models.networks.embedders.composed_xyzt_embedder import ComposedXyztEmbedder
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder

from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor
from easyvolcap.models.networks.regressors.zero_regressor import ZeroRegressor
from easyvolcap.models.networks.regressors.noop_regressor import NoopRegressor
from easyvolcap.models.networks.regressors.split_regressor import SplitRegressor
from easyvolcap.models.networks.regressors.empty_regressor import EmptyRegressor
from easyvolcap.models.networks.regressors.contract_regressor import ContractRegressor


@NETWORKS.register_module()
class VolumetricVideoNetwork(GradientModule):
    try: types = cfg.runner_cfg.visualizer_cfg.types  # list of visualization types # MARK: global
    except AttributeError as e: types = []  # sometimes the list is empty

    # fmt: off
    def __init__(self,
                 # Embedders
                 xyzt_embedder_cfg: dotdict = dotdict(type=ComposedXyztEmbedder.__name__),  # xyzt -> feat
                 xyz_embedder_cfg:  dotdict = dotdict(type=EmptyEmbedder.__name__),  # canonical space embedder
                 dir_embedder_cfg:  dotdict = dotdict(type=PositionalEncodingEmbedder.__name__, multires=4),  # feat for appearance only
                 rgb_embedder_cfg:  dotdict = dotdict(type=EmptyEmbedder.__name__),  # use in conjucture with dir_embedder_cfg

                 # Regressors
                 parameterizer_cfg: dotdict = dotdict(type=ContractRegressor.__name__, in_dim=3),  # xyztθφ -> xyztθφ
                 deformer_cfg:      dotdict = dotdict(type=EmptyRegressor.__name__),  # deformation regressor
                 geometry_cfg:      dotdict = dotdict(type=SplitRegressor.__name__),  # geometry regressor -> occ, feat
                 appearance_cfg:    dotdict = dotdict(type=MlpRegressor.__name__, width=256, depth=2, out_dim=3),  # appearance regressor, rgb

                 # Visualization & Loss
                 train_store_norm: bool = False,
                 train_store_jacobian: bool = False,

                 # Other switches
                 finite_diff: float = -1.0, # finite differentiation for computing normal
                 geo_use_xyzt_feat: bool = True, # only use xyz_feat instead of xyzt + xyz for regression
                 geo_use_xyz_feat: bool = True, # only use xyz_feat instead of xyzt + xyz for regression
                 app_use_geo_feat: bool = True, # which geometry feature to use for the appearance input
                 app_use_xyz_feat: bool = False, # which geometry feature to use for the appearance input
                 app_use_xyzt_feat: bool = False, # which geometry feature to use for the appearance input

                 # Avoiding OOMs
                 chunk_size: int = 524288,
                 ):
        super().__init__()

        # Able to access data created in sampler through batch.output
        # Should return output? (containing all things for loss?)

        # Construct all regressors (possibly networks)
        # For space mapping: mostly space contraction
        self.parameterizer: ContractRegressor          = REGRESSORS.build(parameterizer_cfg) # MARK: hard coded

        # Construct all embedders (possibly networks)
        self.xyzt_embedder: ComposedXyztEmbedder       = EMBEDDERS.build(xyzt_embedder_cfg) # xyz(pe) + latent code
        self.xyz_embedder:  EmptyEmbedder              = EMBEDDERS.build(xyz_embedder_cfg)
        self.dir_embedder:  PositionalEncodingEmbedder = EMBEDDERS.build(dir_embedder_cfg)
        self.rgb_embedder:  EmptyEmbedder              = EMBEDDERS.build(rgb_embedder_cfg)

        # For deformation, residual vector based or se3 field based
        deformer_in_dim                                = self.xyzt_embedder.out_dim
        self.deformer:      EmptyRegressor             = REGRESSORS.build(deformer_cfg,      in_dim=deformer_in_dim)

        # For regressing the geometry (output occupancy), could be implemented as density or just output occ
        geometry_in_dim                                = 0
        if geo_use_xyz_feat:                           geometry_in_dim += self.xyz_embedder.out_dim
        if geo_use_xyzt_feat:                          geometry_in_dim += self.xyzt_embedder.out_dim
        self.geometry:      SplitRegressor             = REGRESSORS.build(geometry_cfg,      in_dim=geometry_in_dim)

        # For regressing the appearance (output 3-channel rgb), could be implemented by IBR, use different kinds of input features
        appearance_in_dim                              = self.dir_embedder.out_dim + self.rgb_embedder.out_dim
        if app_use_geo_feat:                           appearance_in_dim += self.geometry.out_dim - 1
        if app_use_xyz_feat:                           appearance_in_dim += self.xyz_embedder.out_dim
        if app_use_xyzt_feat:                          appearance_in_dim += self.xyzt_embedder.out_dim
        self.appearance:    MlpRegressor               = REGRESSORS.build(appearance_cfg,    in_dim=appearance_in_dim)

        # Misc visualization config
        self.train_store_norm                          = train_store_norm
        self.train_store_jacobian                      = train_store_jacobian
        self.finite_diff                               = finite_diff
        self.geo_use_xyzt_feat                         = geo_use_xyzt_feat
        self.geo_use_xyz_feat                          = geo_use_xyz_feat
        self.app_use_geo_feat                          = app_use_geo_feat
        self.app_use_xyzt_feat                         = app_use_xyzt_feat
        self.app_use_xyz_feat                          = app_use_xyz_feat

        # Avoiding OOM
        self.chunk_size = chunk_size
        self.forward = self.compute # for consistent api
    # fmt: on

    @property
    def chunk_size(self):
        if hasattr(self.xyzt_embedder, 'chunk_size'): return self.xyzt_embedder.chunk_size
        if hasattr(self.xyz_embedder, 'chunk_size'): return self.xyz_embedder.chunk_size
        if hasattr(self.dir_embedder, 'chunk_size'): return self.dir_embedder.chunk_size
        if hasattr(self.rgb_embedder, 'chunk_size'): return self.rgb_embedder.chunk_size

        if hasattr(self.deformer, 'chunk_size'): return self.deformer.chunk_size
        if hasattr(self.geometry, 'chunk_size'): return self.geometry.chunk_size
        if hasattr(self.appearance, 'chunk_size'): return self.appearance.chunk_size
        return -1

    @chunk_size.setter
    def chunk_size(self, chunk_size: int):
        if hasattr(self.xyzt_embedder, 'chunk_size'): self.xyzt_embedder.chunk_size = chunk_size
        if hasattr(self.xyz_embedder, 'chunk_size'): self.xyz_embedder.chunk_size = chunk_size
        if hasattr(self.dir_embedder, 'chunk_size'): self.dir_embedder.chunk_size = chunk_size
        if hasattr(self.rgb_embedder, 'chunk_size'): self.rgb_embedder.chunk_size = chunk_size

        if hasattr(self.deformer, 'chunk_size'): self.deformer.chunk_size = chunk_size
        if hasattr(self.geometry, 'chunk_size'): self.geometry.chunk_size = chunk_size
        if hasattr(self.appearance, 'chunk_size'): self.appearance.chunk_size = chunk_size

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        from imgui_bundle import imgui
        chunk_size = self.chunk_size
        if chunk_size > 0:
            self.chunk_size = imgui.slider_int('In-network chunk size', chunk_size, 512, 1048576)[1]

    @property
    def store_norm(self): return (self.train_store_norm and self.training) or (Visualization.NORMAL.name in VolumetricVideoNetwork.types and not self.training)
    @property
    def store_jacobian(self): return (self.train_store_jacobian and self.training) or (Visualization.JACOBIAN.name in VolumetricVideoNetwork.types and not self.training)

    def compute_geometry(self,
                         xyz: torch.Tensor,
                         t: torch.Tensor,
                         dist: torch.Tensor = 0.005,
                         batch: dotdict = None):

        # Preprocessing
        if self.store_norm or self.store_jacobian:
            xyz.requires_grad_(True)
            grad_ctx_mgr = torch.enable_grad()
            grad_ctx_mgr.__enter__()

        # Pass space conversion
        param_xyz = self.parameterizer(xyz, batch)

        # Pass 4d embedder
        xyzt_feat = self.xyzt_embedder(param_xyz, t, batch)  # find previous output in batch.output

        # Pass deformation
        resd: torch.Tensor = self.deformer(xyzt_feat, batch)  # (B, P, 6)
        if resd.shape[-1] == 6: resd_xyz = apply_rt(param_xyz, resd)  # MARK: this might be heavy
        elif resd.shape[-1] == 3: resd_xyz = param_xyz + resd
        elif resd.shape[-1] == 0: resd_xyz = param_xyz  # no residual
        else: raise NotImplementedError
        # https://stackoverflow.com/questions/16706956/is-there-a-difference-between-raise-exception-and-raise-exception-without

        # Pass 3d embedder
        xyz_feat = self.xyz_embedder(resd_xyz, batch)

        # Pass geometry decoder
        geo_in_feat = []
        if self.geo_use_xyzt_feat: geo_in_feat.append(xyzt_feat)
        if self.geo_use_xyz_feat: geo_in_feat.append(xyz_feat)
        geo_in_feat = torch.cat(geo_in_feat, dim=-1)
        geometry: torch.Tensor = self.geometry(geo_in_feat, batch)

        # https://github.com/pytorch/pytorch/issues/27336
        # Still waiting on update from pytorch to avoid this awkward splitting
        density, geo_feat = geometry.split([1, geometry.shape[-1] - 1], dim=-1)
        occ = raw2alpha(density, dist)  # this might make importance sampling not work!

        # Post processing
        if self.store_norm: norm = normalize(-self.take_gradient(occ, xyz))
        if self.store_jacobian: jacobian = self.take_jacobian(resd_xyz, param_xyz)
        if 'grad_ctx_mgr' in locals(): grad_ctx_mgr.__exit__(None, None, None)

        # Main output
        output = dotdict()
        if resd.shape[-1] == 6: output.rt = resd
        elif resd.shape[-1] == 3: output.resd = resd
        elif resd.shape[-1] == 0: pass
        else: raise NotImplementedError
        output.occ = occ
        output.xyz = xyz  # remember input xyz
        output.t = t
        output.density = density
        output.xyz_feat = xyz_feat
        output.geo_feat = geo_feat
        output.xyzt_feat = xyzt_feat
        output.param_xyz = param_xyz
        output.resd_xyz = resd_xyz

        # Extra output (requires heavy computation)
        if self.store_norm: output.norm = norm
        if self.store_jacobian: output.jacobian = jacobian
        return output

    def compute_appearance(self,
                           dir: torch.Tensor,
                           t: torch.Tensor,  # raw
                           geo_feat: torch.Tensor,  # intermediate
                           batch: dotdict = None):
        # Pass dir embedder
        dir_feat = self.dir_embedder(dir, batch)

        # Pass rgb embedder
        rgb_feat = self.rgb_embedder(t, batch)  # mostly temporal

        # Pass rgb decoder
        app_feat = torch.cat([geo_feat, dir_feat, rgb_feat], dim=-1)
        rgb = self.appearance(app_feat, batch)

        # Main output
        output = dotdict()
        output.rgb = rgb
        output.app_feat = app_feat
        output.dir_feat = dir_feat
        return output

    def occ(self,
            xyz: torch.Tensor,
            t: torch.Tensor,
            dist: torch.Tensor = 0.005,
            batch: dotdict = None) -> torch.Tensor:
        return self.compute_geometry(xyz, t, dist, batch).occ

    def resd_xyz(self, xyz: torch.Tensor, t: torch.Tensor, batch: dotdict):
        # Pass space conversion
        param_xyz = self.parameterizer(xyz, batch)
        # Pass 4d embedder
        xyzt_feat = self.xyzt_embedder(param_xyz, t, batch)  # find previous output in batch.output
        # Pass deformation
        resd: torch.Tensor = self.deformer(xyzt_feat, batch)  # (B, P, 6)
        if resd.shape[-1] == 6: resd_xyz = apply_rt(param_xyz, resd)  # MARK: this might be heavy
        elif resd.shape[-1] == 3: resd_xyz = param_xyz + resd
        else: raise NotImplementedError
        return resd_xyz

    def density(self,
                xyz: torch.Tensor,
                t: torch.Tensor,
                dist: torch.Tensor = 0.005,
                batch: dotdict = None):
        return self.compute_geometry(xyz, t, dist, batch).density

    def gradient(self,
                 xyz: torch.Tensor, t: torch.Tensor,
                 batch: dotdict):
        if self.finite_diff > 0:
            density = self.occ(xyz, t, batch)
            px = xyz.clone()  # is this differentiable?1
            py = xyz.clone()
            pz = xyz.clone()
            px[..., 0] += self.finite_diff
            py[..., 1] += self.finite_diff
            pz[..., 2] += self.finite_diff
            return torch.cat([self.occ(px, t, batch=batch) - density, self.occ(px, t, batch=batch) - density, self.occ(px, t, batch=batch) - density], dim=-1) / self.finite_diff
        else:
            xyz = xyz.requires_grad_()  # inplace modified
            with torch.enable_grad():
                density = self.occ(xyz, t, batch=batch)
            return self.take_gradient(density, xyz)

    def normal(self,
               xyz: torch.Tensor, t: torch.Tensor,
               batch: dotdict):
        return normalize(-self.gradient(xyz, t, batch=batch))

    def compute(self,
                xyz: torch.Tensor, dir: torch.Tensor, t: torch.Tensor, dist: torch.Tensor,
                batch: dotdict):
        # xyz: B, P, 3
        # dir: B, P, 3
        # t: B, P, 1
        # batch: dotdict
        # output: dotdict, output from sampler, should integrate on this

        # This pipeline should cover most of the cases
        # So try to only change its inner part instead of the whole pipeline
        # Unless you're looking for doing funny things like stream training or meta-learning

        # Forward pass for the network components
        output = dotdict()

        # Forward geometry model (prepare most of the features)
        output.update(self.compute_geometry(xyz, t, dist, batch))

        # Prepare appearance input features
        app_in_feat = []
        if self.app_use_geo_feat: app_in_feat.append(output.geo_feat)
        if self.app_use_xyzt_feat: app_in_feat.append(output.xyzt_feat)
        if self.app_use_xyz_feat: app_in_feat.append(output.xyz_feat)
        app_in_feat = torch.cat(app_in_feat, dim=-1) if len(app_in_feat) >= 1 else torch.empty(*(xyz[..., :0].shape), device=xyz.device)

        # Forward appearance model
        output.update(self.compute_appearance(dir, t, app_in_feat, batch))
        rgb, occ = output.rgb, output.occ

        # Store everything for now
        batch.output.update(output)

        # Main output (should only use forward as interface in most cases)
        return rgb, occ
