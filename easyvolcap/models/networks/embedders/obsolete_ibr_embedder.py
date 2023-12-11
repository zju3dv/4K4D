import torch
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.networks.embedders.image_based_embedder import ImageBasedEmbedder

from easyvolcap.utils.data_utils import to_x
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.mvs_utils import sample_feature_image, FeatureAggIBRNet


@EMBEDDERS.register_module()
class ObsoleteIbrEmbedder(ImageBasedEmbedder):
    def __init__(self,
                 agg_cfg: dotdict = dotdict(type=FeatureAggIBRNet.__name__),
                 use_vox_feat: bool = False,
                 use_img_feat: bool = True,
                 use_mvs_dens: bool = True,
                 vox_dim: int = 8,  # 
                 src_dim: int = 32 + 3,  # , 32 image fature dim + 3 rgb dim
                 dtype: str = 'float',
                 ) -> None:
        super().__init__()
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        assert use_vox_feat or use_img_feat, 'At least use some feature'
        self.use_vox_feat = use_vox_feat
        self.use_img_feat = use_img_feat
        self.use_mvs_dens = use_mvs_dens

        self.agg = REGRESSORS.build(agg_cfg, feat_ch=src_dim, use_mvs_dens=use_mvs_dens).to(self.dtype)
        self.out_dim = 1  # 1 computed `density`, no `geo_feat`

    def forward(self, xyz: torch.Tensor, batch: dotdict):
        # If the xyz features are not present, perform sampling
        output = dotdict()

        # Extract things from batch
        src_scale = batch.output.src_scale  # to fix padding issue
        correct_pix = batch.output.correct_pix  # to decide how to generate gird for interpolation
        src_inps, src_feat = batch.output.src_inps, batch.output.src_feat  # (B, S, 3, Hp, Wp), (B, S, C, Hs, Ws)
        del batch.output.src_scale, batch.output.src_inps, batch.output.src_feat, batch.output.feat_vol  # `feat_vol` is not used in `IbrEmbedder`
        tar_ext, _, src_exts, src_ixts = to_x([batch.tar_ext, batch.tar_ixt, batch.src_exts, batch.src_ixts], self.dtype)

        # Sample image feature, source image rgb, source image feature, view direction
        src_msk_feat_rgb_dir = sample_feature_image(xyz, src_feat, src_inps, tar_ext, src_exts, src_scale, src_ixts, correct_pix)  # (B, S, P*D, 1+C+3+4)
        # Aggregate image feature, removed in haotong's latest `htcode` https://github.com/haotongl/htcode/blob/main/lib/networks/nerf/fields/ibrnet_net.py
        density, app_feat = self.agg(src_msk_feat_rgb_dir)  # (B, S, P*D, C) -> (B, P*D, S, C), C == 37, hard coded by `NewAggFeature`

        # Use `density` estimated by cost volume in `CVVRPSampler.prepare_mvs_output()` directly if `self.use_mvs_dens` is set
        density = batch.output.density[..., None].reshape(src_inps.shape[0], -1, 1) if self.use_mvs_dens else density  # (B, P*D, 1)

        # Store `src_msk_rgb` and `app_feat` for appearance regressor
        output.src_msk_rgb = torch.cat([src_msk_feat_rgb_dir[..., :1], src_msk_feat_rgb_dir[..., -4-3:-4]], dim=-1)  # (B, S, P*D, 1+3)
        output.app_feat = app_feat.permute(0, 2, 1, 3)  # (B, S, P*D, C), C = 37
        batch.output.update(output)

        # Return computed `denisty` directly, no any geometry feature, which means we do not perform any real
        # geometry regression in `VolumetricVideoNetwork`, and will not provide `geo_feat` for appearance regressor,
        # the needed appearance features are stored in `batch.output.app_feat`
        return density
