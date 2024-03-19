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

import torch
from torch import nn
import torch.nn.functional as F
from easyvolcap.engine import cfg
from easyvolcap.engine import NETWORKS, EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import Visualization
from easyvolcap.utils.net_utils import GradientModule
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.blend_utils import apply_rt

# Only for type annotation and default arguments
# Isn't this a little bit clumsy? Retyping the network name everytime?
from easyvolcap.models.networks.embedders.empty_embedder import EmptyEmbedder
from easyvolcap.models.networks.embedders.composed_xyzt_embedder import ComposedXyztEmbedder
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder

from easyvolcap.models.networks.regressors.neus_regressor import SDFRegressor, ColorRegressor, SingleVarianceRegressor
from easyvolcap.models.networks.regressors.empty_regressor import EmptyRegressor
from easyvolcap.models.networks.regressors.contract_regressor import ContractRegressor


@NETWORKS.register_module()
class NeuSNetwork(GradientModule):
    try: types = cfg.runner_cfg.visualizer_cfg.types
    except AttributeError as e: types = []

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
                 geometry_cfg:      dotdict = dotdict(type=SDFRegressor.__name__),  # geometry regressor -> occ, feat
                 appearance_cfg:    dotdict = dotdict(type=ColorRegressor.__name__, width=256, depth=2, out_dim=3),  # appearance regressor, rgb
                 deviation_cfg:     dotdict = dotdict(type=SingleVarianceRegressor.__name__, init_val=0.3),

                 # Other switches
                 geo_use_xyzt_feat: bool = False, # only use xyz_feat instead of xyzt + xyz for regression
                 geo_use_xyz_feat: bool = True, # only use xyz_feat instead of xyzt + xyz for regression
                 app_use_geo_feat: bool = True, # which geometry feature to use for the appearance input
                 app_use_sdf_feat: bool = True, # which geometry feature to use for the appearance input
                 app_use_grad_feat: bool = True, # which geometry feature to use for the appearance input
                 app_use_xyz_feat: bool = False, # which geometry feature to use for the appearance input
                 app_use_xyzt_feat: bool = False, # which geometry feature to use for the appearance input

                 # Hash configs mode
                 hash_cfg: str = 'xyz',

                 # Finite difference
                 use_finite_diff: bool = False,
                 use_finite_diff_schedule: bool = False,

                 # Hash schedule 
                 use_hash_encoding_schedule: bool = False,
                 level_init: int = 4,
                 steps_per_level: int = 5000,

                 # Curvature loss schedule
                 use_curvature_loss_weight_schedule: bool = False,
                 curvature_loss_warmup_steps: int = 5000,

                 **kwargs, # ignore other entries
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
        geometry_in_dim                                = 3
        if geo_use_xyz_feat:                           geometry_in_dim += self.xyz_embedder.out_dim
        if geo_use_xyzt_feat:                          geometry_in_dim += self.xyzt_embedder.out_dim
        self.geometry:      SDFRegressor               = REGRESSORS.build(geometry_cfg,      in_dim=geometry_in_dim)

        # For regressing the appearance (output 3-channel rgb), could be implemented by IBR, use different kinds of input features
        appearance_in_dim                              = self.dir_embedder.out_dim + self.rgb_embedder.out_dim + 3
        if app_use_geo_feat:                           appearance_in_dim += self.geometry.out_dim - 1
        if app_use_sdf_feat:                           appearance_in_dim += 1
        if app_use_grad_feat:                          appearance_in_dim += 3
        if app_use_xyz_feat:                           appearance_in_dim += self.xyz_embedder.out_dim
        if app_use_xyzt_feat:                          appearance_in_dim += self.xyzt_embedder.out_dim
        self.appearance:    ColorRegressor             = REGRESSORS.build(appearance_cfg,    in_dim=appearance_in_dim)

        # For regressing the deviation
        self.deviation:     SingleVarianceRegressor    = REGRESSORS.build(deviation_cfg)

        # Misc visualization config
        self.geo_use_xyzt_feat                         = geo_use_xyzt_feat
        self.geo_use_xyz_feat                          = geo_use_xyz_feat
        self.app_use_geo_feat                          = app_use_geo_feat
        self.app_use_sdf_feat                          = app_use_sdf_feat
        self.app_use_grad_feat                         = app_use_grad_feat
        self.app_use_xyzt_feat                         = app_use_xyzt_feat
        self.app_use_xyz_feat                          = app_use_xyz_feat

        self.use_hash_encoding_schedule                = use_hash_encoding_schedule
        self.steps_per_level                           = steps_per_level
        self.level_init                                = level_init

        if hash_cfg == 'xyz' and 'Hash' in xyz_embedder_cfg.type:
            self.hash_cfg = xyz_embedder_cfg
        elif hash_cfg == 'xyzt' and 'Hash' in xyzt_embedder_cfg.type:
            self.hash_cfg = xyzt_embedder_cfg.xyz_embedder_cfg
        else:
            self.hash_cfg = None
        if self.hash_cfg is not None:
            self.base_resolution                       = self.hash_cfg.base_resolution
            self.b                                     = self.hash_cfg.b
            self.num_levels                            = self.hash_cfg.n_levels
            self.max_resolution                        = int(self.b ** (self.num_levels - 1) * self.base_resolution)

        self.use_finite_diff                           = use_finite_diff
        self.finite_diff_delta : float                 = 0.0001
        self.use_finite_diff_schedule                  = use_finite_diff_schedule

        self.use_curvature_loss_weight_schedule        = use_curvature_loss_weight_schedule
        self.curvature_loss_multi_factor: float        = 1.0
        self.curvature_loss_warmup_steps                = curvature_loss_warmup_steps

        self.anneal_end: int                           = 50000
        self.cos_anneal_ratio: float                   = 1.0
        self.compute = self.forward
    # fmt: on

    def export_mesh(self):
        from easyvolcap.utils.mesh_utils import get_surface_sliding
        def sdf(x): return self.sdf(x, torch.zeros_like(x[..., :1]), None,
                                    skip_parameterization=True, skip_deformation=True)[..., 0]
        get_surface_sliding(
            sdf,
            bounding_box_min=[-1, -1, -1],
            bounding_box_max=[1, 1, 1],
            resolution=256)

    def set_cos_anneal_ratio(self, anneal: float):
        self.cos_anneal_ratio = anneal

    def set_finite_diff_delta(self, delta: float):
        self.finite_diff_delta = delta

    def set_curvature_loss_mutli_factor(self, factor: float):
        self.curvature_loss_multi_factor = factor

    def raw2alpha(self,
                  dir: torch.Tensor,
                  sdf: torch.Tensor,
                  gradient: torch.Tensor,
                  dist: torch.Tensor):

        inv_s = self.deviation.get_variance()

        true_cos = (dir * gradient).sum(dim=-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self.cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dist * 0.5
        estimated_prev_sdf = sdf - iter_cos * dist * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def compute_geometry(self,
                         xyz: torch.Tensor,
                         t: torch.Tensor,
                         batch: dotdict,
                         compute_grad: bool = False,
                         compute_jacob: bool = False,
                         skip_parameterization: bool = False,
                         skip_deformation: bool = False,
                         return_sdf_only: bool = False):

        # Pass space conversion
        if skip_parameterization:
            param_xyz = xyz
        else:
            param_xyz = self.parameterizer(xyz, batch)
        param_xyz_norm = torch.norm(param_xyz, dim=-1)

        if compute_jacob:
            param_xyz.requires_grad_(True)
            grad_ctx_mgr = torch.enable_grad()
            grad_ctx_mgr.__enter__()

        # Pass 4d embedder
        xyzt_feat = self.xyzt_embedder(param_xyz, t, batch)  # find previous output in batch.output

        # Pass deformation
        if skip_deformation:
            resd_xyz = param_xyz
        else:
            resd: torch.Tensor = self.deformer(xyzt_feat, batch)  # (B, P, 6)
            if resd.shape[-1] == 6: resd_xyz = apply_rt(param_xyz, resd)  # MARK: this might be heavy
            elif resd.shape[-1] == 3: resd_xyz = param_xyz + resd
            elif resd.shape[-1] == 0: resd_xyz = param_xyz
            else: raise NotImplementedError
            # https://stackoverflow.com/questions/16706956/is-there-a-difference-between-raise-exception-and-raise-exception-without

        if (not compute_jacob) and compute_grad:
            resd_xyz.requires_grad_(True)
            grad_ctx_mgr = torch.enable_grad()
            grad_ctx_mgr.__enter__()
            # Check test_using_inference_mode if the gradient manager doesn't work

        # Pass 3d embedder
        xyz_feat = self.xyz_embedder(resd_xyz, batch)

        # Pass geometry decoder
        geo_in_feat = [resd_xyz]
        if self.geo_use_xyzt_feat: geo_in_feat.append(xyzt_feat)
        if self.geo_use_xyz_feat: geo_in_feat.append(xyz_feat)
        geo_in_feat = torch.cat(geo_in_feat, dim=-1)
        geometry: torch.Tensor = self.geometry(geo_in_feat, batch)

        # https://github.com/pytorch/pytorch/issues/27336
        # Still waiting on update from pytorch to avoid this awkward splitting
        sdf, geo_feat = geometry.split([1, geometry.shape[-1] - 1], dim=-1)
        if return_sdf_only: return sdf

        # Post processing
        if compute_grad:
            if self.use_finite_diff:
                gradients, sampled_sdf = self.gradient(resd_xyz.detach(), t, batch, return_sdf=True)
            else:
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = self.take_gradient(sdf, resd_xyz, d_output)
                sampled_sdf = None
        if compute_jacob:
            jacobian = self.take_jacobian(resd_xyz, param_xyz)
        if 'grad_ctx_mgr' in locals(): grad_ctx_mgr.__exit__(None, None, None)

        # Main output
        output = dotdict()
        if resd.shape[-1] == 6: output.rt = resd
        elif resd.shape[-1] == 3: output.resd = resd
        elif resd.shape[-1] == 0: pass
        else: raise NotImplementedError
        output.sdf = sdf
        output.xyz = xyz  # xyz after transformations NOTE: before?
        output.xyz_feat = xyz_feat
        output.geo_feat = geo_feat
        output.xyzt_feat = xyzt_feat
        output.param_xyz = param_xyz
        output.param_xyz_norm = param_xyz_norm
        output.resd_xyz = resd_xyz

        # Extra output (require heavy computation)
        if compute_grad:
            output.gradients = gradients
            output.norm = normalize(gradients)
            if sampled_sdf is not None:
                output.sampled_sdf = sampled_sdf
        if compute_jacob:
            output.jacobian = jacobian
        return output

    def compute_appearance(self,
                           xyz: torch.Tensor,
                           dir: torch.Tensor,
                           t: torch.Tensor,  # raw
                           geo_feat: torch.Tensor,  # intermediate
                           batch: dotdict):
        # Pass dir embedder
        dir_feat = self.dir_embedder(dir, batch)

        # Pass rgb embedder
        rgb_feat = self.rgb_embedder(t, batch)  # mostly temporal

        # Pass rgb decoder
        app_feat = torch.cat([xyz, dir_feat, geo_feat, rgb_feat], dim=-1)
        rgb = self.appearance(app_feat, batch)

        # Main output
        output = dotdict()
        output.rgb = rgb
        output.app_feat = app_feat
        output.dir_feat = dir_feat
        output.rgb_feat = rgb_feat
        return output

    def sdf(self,
            xyz: torch.Tensor,
            t: torch.Tensor,
            batch: dotdict,
            skip_parameterization: bool = False,
            skip_deformation: bool = False) -> torch.Tensor:
        return self.compute_geometry(xyz, t, batch,
                                     compute_grad=False,
                                     compute_jacob=False,
                                     skip_parameterization=skip_parameterization,
                                     skip_deformation=skip_deformation,
                                     return_sdf_only=True)

    def occ(self,
            xyz: torch.Tensor,
            dir: torch.Tensor,
            t: torch.Tensor,
            dist: torch.Tensor,
            batch: dotdict) -> torch.Tensor:
        output = self.compute_geometry(xyz, t, batch, compute_grad=True)
        return self.raw2alpha(dir, output.sdf, output.gradients, dist)

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

    def gradient(self,
                 xyz: torch.Tensor,
                 t: torch.Tensor,
                 batch: dotdict,
                 return_sdf: bool = False):
        # NOTE: compute gradients in canonical space
        # NOTE: xyz in this function should be resd_xyz
        if self.use_finite_diff:
            delta = self.finite_diff_delta
            xyz_shape = xyz.shape  # (B, P, 3)
            assert len(xyz_shape) == 3
            t_shape = t.shape  # (B, P, 1)
            device = xyz.device
            dtype = xyz.dtype
            xyz = torch.stack(
                [
                    xyz + torch.tensor([delta, 0.0, 0.0], dtype=dtype, device=device),
                    xyz + torch.tensor([-delta, 0.0, 0.0], dtype=dtype, device=device),
                    xyz + torch.tensor([0.0, delta, 0.0], dtype=dtype, device=device),
                    xyz + torch.tensor([0.0, -delta, 0.0], dtype=dtype, device=device),
                    xyz + torch.tensor([0.0, 0.0, delta], dtype=dtype, device=device),
                    xyz + torch.tensor([0.0, 0.0, -delta], dtype=dtype, device=device),
                ],
                dim=0,
            ).view(-1, *xyz_shape[1:])  # (6, B, P, 3) -> (-1, P, 3)
            t = torch.repeat_interleave(t[None], 6, dim=0).view(-1, *t_shape[1:])  # (6, B, P, 1) -> (-1, P, 1)
            sdf = self.sdf(xyz, t, batch, skip_parameterization=True, skip_deformation=True)
            sdf = sdf.view(6, *xyz_shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (sdf[0] - sdf[1]) / delta,
                    0.5 * (sdf[2] - sdf[3]) / delta,
                    0.5 * (sdf[4] - sdf[5]) / delta,
                ],
                dim=-1,
            )
            sdf = sdf.permute(1, 2, 0).contiguous()

        else:
            xyz = xyz.requires_grad_(True)  # inplace modified
            with torch.enable_grad():
                sdf = self.sdf(xyz, t, batch, skip_parameterization=True, skip_deformation=True)
            d_out = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = self.take_gradient(sdf, xyz, d_out)

        if return_sdf:
            return gradients, sdf
        else:
            return gradients

    def normal(self,
               xyz: torch.Tensor,
               t: torch.Tensor,
               batch: dotdict):
        return normalize(self.gradient(xyz, t, batch, return_sdf=False))

    def _forward(self,
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
        output.update(self.compute_geometry(xyz, t, batch, compute_grad=True))

        # Compute occ
        output.occ = self.raw2alpha(dir, output.sdf, output.gradients, dist)
        output.deviation = self.deviation.get_variance().view(1, 1)

        # Prepare appearance input features
        app_in_feat = []
        if self.app_use_grad_feat: app_in_feat.append(output.gradients)
        if self.app_use_sdf_feat: app_in_feat.append(output.sdf)
        if self.app_use_geo_feat: app_in_feat.append(output.geo_feat)
        if self.app_use_xyzt_feat: app_in_feat.append(output.xyzt_feat)
        if self.app_use_xyz_feat: app_in_feat.append(output.xyz_feat)
        app_in_feat = torch.cat(app_in_feat, dim=-1) if len(app_in_feat) >= 1 else torch.empty(*(xyz[..., :0].shape), device=xyz.device)

        # Forward appearance model
        output.update(self.compute_appearance(output.resd_xyz, dir, t, app_in_feat, batch))
        rgb, occ = output.rgb, output.occ

        # Store everything for now
        batch.output.update(output)

        # Main output (should only use forward as interface in most cases)
        return rgb, occ

    def forward(self,
                xyz: torch.Tensor,
                dir: torch.Tensor,
                t: torch.Tensor,
                dist: torch.Tensor,
                batch: dotdict):
        self.before_forward(batch)
        rgb, occ = self._forward(xyz, dir, t, dist, batch)
        self.after_forward(batch)
        return rgb, occ

    def before_forward(self, batch: dotdict):
        iter = batch.iter.item()

        if self.anneal_end > 0:
            if self.training:
                anneal = min([1.0, iter / self.anneal_end])
                self.set_cos_anneal_ratio(anneal)

        if self.use_finite_diff_schedule:
            if self.training:
                delta = 1. / (self.base_resolution * self.b ** (iter / self.steps_per_level))
                delta = max(1. / self.max_resolution, delta)
                self.set_finite_diff_delta(delta * 4.0)  # TODO should we consider bounds in hashembder?

        if self.use_hash_encoding_schedule:
            if self.training:
                level = int(iter / self.steps_per_level) + 1
                level = max(level, self.level_init)
                if hasattr(self.xyzt_embedder.xyz_embedder, 'update_mask'):
                    self.xyzt_embedder.xyz_embedder.update_mask(level)
                if hasattr(self.xyz_embedder, 'update_mask'):
                    self.xyz_embedder.update_mask(level)

        if self.use_curvature_loss_weight_schedule:
            if self.training:
                if iter < self.curvature_loss_warmup_steps:
                    factor = iter / self.curvature_loss_warmup_steps
                else:
                    delta = 1. / (self.base_resolution * self.b ** ((iter - self.curvature_loss_warmup_steps) / self.steps_per_level))
                    delta = max(1. / (self.max_resolution * 10.0), delta)
                    factor = delta * self.base_resolution
                self.set_curvature_loss_mutli_factor(factor)

    def after_forward(self, batch: dotdict):
        pass
