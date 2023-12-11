import torch

import numpy as np
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.enerf_utils import sample_feature_volume, sample_feature_image, FeatureAgg, FeatureNet
from easyvolcap.utils.ibr_utils import get_src_inps, get_src_feats, prepare_caches, compute_src_inps, compute_src_feats
from easyvolcap.utils.net_utils import normalize, linear_sampling, s_vals_to_z_vals, z_vals_to_s_vals, ray2xyz, volume_rendering, interpolate_image, resize_image, create_meshgrid, multi_scatter, multi_gather, fill_nhwc_image, fill_nchw_image


@EMBEDDERS.register_module()
class StableENeRFImageBasedEmbedder(nn.Module):
    def __init__(self,
                 agg_cfg: dotdict = dotdict(type=FeatureAgg.__name__),
                 feat_cfg: dotdict = dotdict(type=FeatureNet.__name__),
                 cache_size: int = 512,  # (512 * 512 * 3 * 8 + 256 * 256 * 3 * 16 + 128 * 128 * 3 * 32) * 4 / 2 ** 20 = 42.0 MB -> all cached -> 26 GB of VRAM
                 ) -> None:
        super().__init__()
        self.feat_reg: FeatureNet = REGRESSORS.build(feat_cfg)
        self.src_dim = self.feat_reg.out_dims[-1] + 3 + 4
        self.feat_agg: FeatureAgg = REGRESSORS.build(agg_cfg, feat_ch=self.src_dim)
        self.out_dim = self.feat_agg.out_dim  # 16 or 0
        self.img_pad = self.feat_reg.size_pad
        prepare_caches(cache_size)

    def compute_src_feats(self, batch: dotdict):
        # Compute source image features

        # Prepare inputs and feature
        src_inps = compute_src_inps(batch) if self.training else get_src_inps(batch)  # will inplace upate feature
        # Values to be reused
        # Preparing source scaling (for painless up convolution and skip connections)
        Hc, Wc = src_inps.shape[-2:]  # cropped image size
        Hp, Wp = int(np.ceil(Hc / self.img_pad)) * self.img_pad, int(np.ceil(Wc / self.img_pad)) * self.img_pad  # Input and output should be same in size
        src_inps = interpolate_image(src_inps, size=(Hp, Wp))  # B, S, 3, H, W

        # Preparing source image scaling
        src_scale = src_inps.new_empty(2, 1)
        src_scale[0] = Wp / Wc
        src_scale[1] = Hp / Hc

        # Forward feature extraction
        # `src_feats` is a list of features of shape (B, S, C, H, W) -> (B, S, 32*(2**(-i)), H//4*(2**i), W//4*(2**i))
        src_feats = compute_src_feats(src_inps, self.feat_reg, batch) if self.training else get_src_feats(src_inps, self.feat_reg, batch)
        src_feat_rgb = torch.cat([src_feats[-1], src_inps], dim=-3)  # B, S, C, Hr, Wr
        batch.persistent.src_feat_rgb = src_feat_rgb
        batch.persistent.src_scale = src_scale

    def forward(self, xyz: torch.Tensor, batch: dotdict):
        # xyz: B, P * S, 3

        # Find features inside batch
        # Return sampled features

        # If the xyz features are not present, perform sampling
        if 'src_feat_rgb' not in batch.persistent: self.compute_src_feats(batch)

        # Extract things from batch
        src_feat_rgb = batch.persistent.src_feat_rgb  # last level of source image feature, B, S, C, Hs, Ws
        src_scale = batch.persistent.src_scale
        tar_ext, tar_ixt, src_exts, src_ixts = batch.tar_ext, batch.tar_ixt, batch.src_exts, batch.src_ixts

        # Sample image feature
        src_feat_rgb_dir = sample_feature_image(xyz, src_feat_rgb, tar_ext, src_exts, src_scale, src_ixts)  # B, S, P, C

        # Aggregate image feature
        img_feat = self.feat_agg(src_feat_rgb_dir)  # B, S, P, C -> B, P, C

        # Store output
        batch.output.src_feat_rgb_dir = src_feat_rgb_dir  # B, S, P, C
        return img_feat
