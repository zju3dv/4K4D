import torch

import numpy as np
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.enerf_utils import FeatureAgg, FeatureNet
from easyvolcap.utils.ibr_utils import get_src_inps, get_src_feats, prepare_caches, compute_src_inps, compute_src_feats, sample_feature_volume, sample_geometry_feature_image
from easyvolcap.utils.image_utils import interpolate_image, pad_image
from easyvolcap.utils.data_utils import to_x


@EMBEDDERS.register_module()
class GeometryImageBasedEmbedder(nn.Module):
    def __init__(self,
                 feat_cfg: dotdict = dotdict(type=FeatureNet.__name__),
                 n_track: int = 8,  # only the first 8 images receives gradient through feature regressor
                 use_interpolate: bool = False,  # use interpolate instead of fill for image scaling, a legacy switch
                 ) -> None:
        super().__init__()
        self.n_track = n_track
        self.feat_reg: FeatureNet = REGRESSORS.build(feat_cfg)
        self.img_pad = self.feat_reg.size_pad
        self.src_dim = self.feat_reg.out_dims[-1] + 3  # image feature and rgb color for blending

        self.use_interpolate = use_interpolate

    def compute_src_feats(self, batch: dotdict):
        # Prepare inputs and feature
        src_inps = compute_src_inps(batch).to(self.feat_reg.conv0[0].conv.weight.dtype)
        # Values to be reused
        # Preparing source scaling (for painless up convolution and skip connections)
        Hc, Wc = src_inps.shape[-2:]  # cropped image size
        Hp, Wp = int(np.ceil(Hc / self.img_pad)) * self.img_pad, int(np.ceil(Wc / self.img_pad)) * self.img_pad  # Input and output should be same in size
        if self.use_interpolate:
            src_inps = interpolate_image(src_inps, size=(Hp, Wp))  # B, S, 3, H, W
        else:
            src_inps = pad_image(src_inps, size=(Hp, Wp))  # B, S, 3, H, W

        # Preparing source image scaling
        if self.use_interpolate:
            src_scale = src_inps.new_empty(2, 1)
            src_scale[0] = Wp / Wc
            src_scale[1] = Hp / Hc
        else:
            src_scale = src_inps.new_ones(2, 1)  # 2, 1

        # Forward feature extraction
        # `src_feats` is a list of features of shape (B, S, C, H, W) -> (B, S, 32*(2**(-i)), H//4*(2**i), W//4*(2**i))
        feats = []
        for i, inp in enumerate(src_inps[0]):  # !: BATCH
            if i < self.n_track:
                feats.append(compute_src_feats(inp, self.feat_reg, batch)[-1])  # C, H, W
            else:
                with torch.no_grad():  # no gradient tracking for these images, to save training time
                    feats.append(compute_src_feats(inp, self.feat_reg, batch)[-1])
        src_feat = torch.stack(feats)[None]  # S, C, H, W -> B, S, C, H, W
        src_feat_rgb = torch.cat([src_feat, src_inps], dim=-3)  # B, S, C, Hr, Wr
        batch.persistent.src_feat_rgb = src_feat_rgb
        batch.persistent.src_scale = src_scale

    def forward(self, xyz: torch.Tensor, batch: dotdict, optimize_cnn: bool = True):
        # xyz: B, P * S, 3
        if not optimize_cnn:
            with torch.no_grad():
                self.compute_src_feats(batch)
        else:
            self.compute_src_feats(batch)
        # Extract things from batch
        src_feat_rgb = batch.persistent.src_feat_rgb  # last level of source image feature, B, S, C, Hs, Ws
        src_scale = batch.persistent.src_scale  # maybe use all source input images for feature caching?
        src_exts, src_ixts = to_x([batch.src_exts, batch.src_ixts], xyz.dtype)  # the source camera parameter is still required to perform the sampling of nearby points...

        # Sample image feature
        src_feat_rgb = sample_geometry_feature_image(xyz, src_feat_rgb, src_exts, src_ixts, src_scale)  # B, S, P, C

        # Store output to batch variable, this module is not intended to be inserted as a normal regressor
        batch.output.src_feat_rgb = src_feat_rgb  # B, S, P, C
        return src_feat_rgb  # not to intendied to be used directly # MARK: No dir needed
