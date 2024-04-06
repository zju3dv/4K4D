import torch
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.chunk_utils import chunkify
from easyvolcap.utils.net_utils import MLP, get_function


@REGRESSORS.register_module()
class ImageBasedRegressor(nn.Module):
    def __init__(self,
                 in_dim: int = 0,  # feature channel dim (vox + img?)
                 src_dim: int = 32 + 3 + 4,  # (feat, rgb, dir)

                 width: int = 64,
                 depth: int = 1,  # use small regressor network

                 use_dir: bool = True,
                 manual_chunking: bool = False,
                 dtype: str = torch.float,

                 chunk_size: int = 1e20,
                 out_actvn: nn.Module = nn.Identity(),

                 **kwargs,
                 ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.use_dir = use_dir
        self.out_actvn = get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn
        self.manual_chunking = manual_chunking

        self.mlp = MLP(in_dim + src_dim, width, depth, 1, out_actvn=nn.Identity(), dtype=dtype)  # vox(8) + img(16) + geo(64) + img_feat_rgb_dir (32 + 3 + 4)
        self.chunk_size = chunk_size
        # self.forward_mlp = chunkify(chunk_size=chunk_size)(self.mlp)

    def forward(self, geo_feat: torch.Tensor, batch: dotdict):
        # geo_feat: B, P, C # vox(8) + img(16) + geo(64)?

        # Prepare for directional feature
        if self.use_dir:
            src_feat: torch.Tensor = batch.output.src_feat_rgb_dir  # B, S, P, C
            src_rgbs = src_feat[..., -4 - 3:-4]  # -4: dir feat, -7 -> -3: rgb, B, S, P, 3
        else:
            src_feat: torch.Tensor = batch.output.src_feat_rgb
            src_rgbs = src_feat[..., -3:]

        # Prepare for image based rendering blending (in a narrow sense)
        B, S, P, _ = src_feat.shape
        forward_mlp = chunkify(chunk_size=self.chunk_size)(self.mlp)

        # Manual chunking, hopefully this will ease the memory usage
        if self.manual_chunking:
            rgb_bws = []
            for i in range(B):  # TODO: PERF
                for j in range(S):
                    feat = torch.cat([geo_feat[i], src_feat[i, j]], dim=-1)  # P, C
                    rgb_bws.append(forward_mlp(feat))  # P, 1
            rgb_bws = torch.stack(rgb_bws)  # BS, P, 1
            rgb_bws = rgb_bws.view(B, S, P, -1)  # B, S, P, 1
        else:
            geo_feat = geo_feat[:, None].expand(src_feat.shape[:-1] + (geo_feat.shape[-1], ))  # B, S, P, C
            geo_feat = torch.cat([geo_feat, src_feat], dim=-1)  # +7, append the actual image feature
            del src_feat  # release some memory
            rgb_bws = forward_mlp(geo_feat)

        rgb_bws = rgb_bws.softmax(-3)  # B, S, P, 1
        rgb = (src_rgbs * rgb_bws).sum(-3)  # B, P, 3
        return self.out_actvn(rgb)  # this regressor only produces rgb of enerf for now, need more care for reusing
