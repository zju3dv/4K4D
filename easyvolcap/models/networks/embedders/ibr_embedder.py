import torch
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.networks.regressors.noop_regressor import NoopRegressor
from easyvolcap.models.networks.embedders.image_based_embedder import ImageBasedEmbedder

from easyvolcap.utils.data_utils import to_x
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.enerf_utils import FeatureAgg
from easyvolcap.utils.mvs_utils import sample_feature_image, sample_feature_volume, FeatureAggIBRNet

# In order to make this embedder more general, we need to consider all possible output cases:
# 1. [*] `geo_in_feat` and `src_feat`, where `geo_in_feat` is used as final geometry feature in `VolumetricVideoNetwork.compute_geometry()` to decode `density` and `geo_feat`, and `src_feat` is used as part of appearance feature in `IbrEmbedder.forward()` to decode `rgb`;
# 2. [*] `geo_in_feat` and `app_feat`, where `geo_in_feat` is used as final geometry feature in `VolumetricVideoNetwork.compute_geometry()` to decode `density` only, and `app_feat` is used as final appearance feature in `IbrEmbedder.forward()` to decode `rgb`;
# 3. [*] `density` and `src_feat`, in cases where we already have `density` computed before or computed here, and `src_feat` is used as part of appearance feature in `IbrEmbedder.forward()` to decode `rgb`;
# 3. [*] `density` and `app_feat`, in cases where we already have `density` computed before or computed here, and `app_feat` is used as final appearance feature in `IbrEmbedder.forward()` to decode `rgb`;
# Therefore, we organize the output format of this embedder as: explicitly return geometry related variables, and implicitly store appearance related variables in `batch.output.app_feat` for appearance regressor to use.

@EMBEDDERS.register_module()
class IbrEmbedder(ImageBasedEmbedder):
    def __init__(self,
                 agg_cfg: dotdict = dotdict(type=FeatureAgg.__name__),
                 app_cfg: dotdict = dotdict(type=NoopRegressor.__name__),

                 use_vox_feat: bool = False,
                 use_img_feat: bool = True,

                 project_mask: bool = False,  # only used by `IBRNet` agg, `ENeRF` agg does not use this
                 have_density: bool = False,  # whether we already have `density` computed before or computed here
                 cat_ibr_feat: bool = False,  # whether to concatenate `ibr_feat` with `app_feat`, to distinguish from ENeRF and IBRNet for now

                 vox_dim: int = 8,  # 
                 src_dim: int = 32 + 3,  # , 32 image fature dim + 3 rgb dim
                 dtype: str = 'float',
                 ) -> None:
        super().__init__()
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        assert use_vox_feat or use_img_feat, 'At least use some feature'
        self.use_vox_feat = use_vox_feat
        self.use_img_feat = use_img_feat
        self.project_mask = project_mask  # only used by `IBRNet` agg, `ENeRF` agg does not use this
        self.have_density = have_density  # whether we already have `density` computed before or computed here
        self.cat_ibr_feat = cat_ibr_feat  # whether to concatenate `ibr_feat` with `app_feat`, to distinguish from ENeRF and IBRNet for now

        # This `agg` network may have many different output when using different methods, including:
        # [ENeRF]: `geo_in_feat` of shape (B, P*D, C), may used as input geometry feature in `VolumetricVideoNetwork.compute_geometry()` or part of appearance feature in `IbrEmbedder.forward()`;
        # [IBRNet]: `geo_in_feat` of shape (B, P*D, S, C) and `app_feat` of shape (B, P*D, S, C), used as input geometry feature in `VolumetricVideoNetwork.compute_geometry()` and appearance feature in `IbrEmbedder.forward()` respectively;
        self.agg = REGRESSORS.build(agg_cfg, feat_ch=src_dim).to(self.dtype)
        self.app = REGRESSORS.build(app_cfg).to(self.dtype)

        self.out_dim = 0
        if self.use_vox_feat: self.out_dim += vox_dim  # 8
        if self.use_img_feat: self.out_dim += self.agg.out_dim  # 16
        if self.have_density: self.out_dim = 1

    def prepare_geometry(self, ibr_feat: torch.Tensor, batch: dotdict):
        # Pass `ibr_feat` to `self.agg` to get output, may including following possible circumstances:
        # 1. [*] `geo_feat`, which is used as `geo_in_feat` in `VolumetricVideoNetwork.compute_geometry()` to decode `density` and `geo_feat`;
        # 2. [*] `geo_feat`, which is actually used as part of appearance feature in `IbrEmbedder.forward()` to decode `rgb`, the `density` is already computed in the sampler;
        # 3. [*] `[density, app_feat]`, where `density` is used as final `density` and `app_feat` is used as final appearance feature in `IbrEmbedder.forward()` to decode `rgb`;
        # 4. [*] `[None, app_feat]`, where `app_feat` is used as final appearance feature in `IbrEmbedder.forward()` to decode `rgb`, the `density` is already computed in the sampler;
        if self.use_img_feat: img_feat = self.agg(ibr_feat)  # (B, P*D, C) or a list
        if self.use_vox_feat: vox_feat = sample_feature_volume(batch.output.s_vals, batch.output.uv, batch.output.feat_vol)  # (B, P*D, C)

        # First, determine whether `self.agg` outputs only `geo_feat` or both `geo_feat` and `app_feat`
        if isinstance(img_feat, list): geo_feat, app_feat = img_feat  # (B, P*D, C), (B, S, P*D, C)
        elif self.use_img_feat and self.use_vox_feat: geo_feat, app_feat = torch.cat([img_feat, vox_feat], dim=-1), None  # (B, P*D, C), None
        elif self.use_img_feat: geo_feat, app_feat = img_feat, None  # (B, P*D, C), None
        else: geo_feat, app_feat = vox_feat, None  # (B, P*D, C), None

        # Next, determine whether to use the existing `density` or the newly calculated `geo_feat`.
        if self.have_density: geo_feat, app_feat = batch.output.density.reshape(batch.output.B, -1, 1), self.app(geo_feat)  # (B, P*D, 1), (B, P*D, C) or (B, S, P*D, C)

        # Deal with the shape of `app_feat`
        if app_feat is not None and app_feat.ndim < 4: app_feat = app_feat[:, None].expand(-1, ibr_feat.shape[1], -1, -1)  # (B, P*D, C) -> (B, 1, P*D, C) -> (B, S, P*D, C)
        return geo_feat, app_feat

    def prepare_appearance(self, ibr_feat: torch.Tensor, app_feat: torch.Tensor, output: dotdict, batch: dotdict):
        # `ibr_feat` is used as `app_feat` in `IbrEmbedder.forward()` when there is a `geo_feat` output by `VolumeVideoNetwork.compute_geometry()`
        if app_feat is None: app_feat = ibr_feat  # (B, S, P*D, C)
        # concatenate `ibr_feat` with `app_feat` if `self.cat_ibr_feat` is set, `app_feat` here is actually the same `geo_feat` output by `VolumeVideoNetwork.compute_geometry()`
        elif self.cat_ibr_feat: app_feat = torch.cat([ibr_feat, app_feat], dim=-1)  # (B, S, P*D, C)

        # Store `app_feat` and `src_rgb` in `output` for appearance regressor
        output.src_rgbs = ibr_feat[..., -3 - 4:-4]  # (B, S, P*D, 3)
        output.app_feat = app_feat  # (B, S, P*D, C)
        if self.project_mask: output.src_msks = ibr_feat[..., :1]  # (B, S, P*D, 1)
        return output

    def forward(self, xyz: torch.Tensor, batch: dotdict):
        # If the xyz features are not present, perform sampling
        output = dotdict()

        # Extract variables needed for geometry feature preparation from `batch.output`
        src_scale, src_inps, src_feat = batch.output.src_scale, batch.output.src_inps, batch.output.src_feat  # (B, S, 3, Hp, Wp), (B, S, C, Hs, Ws)
        tar_ext, _, src_exts, src_ixts = to_x([batch.tar_ext, batch.tar_ixt, batch.src_exts, batch.src_ixts], self.dtype)
        # Prepare `src_feat_rgb_dir` first, it is used by all methods for now
        src_feat_rgb_dir = sample_feature_image(xyz, src_feat, src_inps, tar_ext, src_exts, src_scale, src_ixts, batch.output.correct_pix, self.project_mask)  # (B, S, P*D, C+3+4) or (B, S, P*D, 1+C+3+4)
        # Delete variables from `batch.output` to save memory, TODO: determine whether it is actually useful
        del batch.output.src_scale, batch.output.src_inps, batch.output.src_feat

        # Prepare geometry features, may be `geo_in_feat`, or `density` already computed in the sampler
        geo_feat, app_feat = self.prepare_geometry(src_feat_rgb_dir, batch)  # (B, P*D, C), (B, S, P*D, C) or None

        # Prepare appearance features, maybe `app_feat` already computed in the sampler
        output = self.prepare_appearance(src_feat_rgb_dir, app_feat, output, batch)
        batch.output.update(output)

        return geo_feat
