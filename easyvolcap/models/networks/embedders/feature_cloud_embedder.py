import torch

from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.fcds_utils import update_features
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder


@EMBEDDERS.register_module()
class FeatureCloudEmbedder(nn.Module):
    def __init__(self,
                 in_dim: int = 64,  # smaller input dim to fit inside the memory
                 radius: float = 0.10,  # larger radius for aggregation
                 K: int = 10,  # otherwise oom

                 xyz_embedder_cfg: dotdict = dotdict(type=PositionalEncodingEmbedder.__name__),
                 ) -> None:
        super().__init__()

        self.in_dim = in_dim

        self.radius = radius
        self.K = K

        self.xyz_embedder = EMBEDDERS.build(xyz_embedder_cfg)
        self.out_dim = in_dim + self.xyz_embedder.out_dim

    def forward(self, xyz: torch.Tensor, batch: dotdict):
        # xyz: B, P * S, 3

        # Find features inside batch
        # Return sampled features

        # This feature is position agnostic
        fcd_feat = update_features(xyz, batch.output.pcd, batch.output.feat, self.radius, self.K)
        xyz_feat = self.xyz_embedder(xyz)
        return torch.cat([fcd_feat, xyz_feat], dim=-1)  # gives more positional information

# @EMBEDDERS.register_module()
# class FeatureCloudEmbedder(nn.Module):
#     def __init__(self,
#                  in_dim: int = 64,  # smaller input dim to fit inside the memory
#                  radius: float = 0.05,  # aggregation
#                  K: int = 5,  # otherwise oom
#                  W: int = 64,
#                  D: int = 2,
#                  ) -> None:
#         super().__init__()

#         self.in_dim = in_dim
#         self.out_dim = in_dim
#         self.aggregator = PointNeRFAggregator(in_dim=in_dim + 3, out_dim=in_dim, K=K, W=W, D=D, radius=radius)  # will be used in aggregation

#     def forward(self, xyz: torch.Tensor, batch: dotdict):
#         # xyz: B, P * S, 3

#         # Find features inside batch
#         # Return sampled features
#         feat = self.aggregator(xyz, batch.output.pcd, batch.output.feat)
#         return feat
