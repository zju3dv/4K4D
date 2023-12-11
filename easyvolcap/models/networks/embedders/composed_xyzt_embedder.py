# This one composes xyzt embedder from an xyz embedder and t embedder

import torch

from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder
from easyvolcap.models.networks.embedders.latent_code_embedder import LatentCodeEmbedder


@EMBEDDERS.register_module()
class ComposedXyztEmbedder(nn.Module):
    def __init__(self,
                 xyz_embedder_cfg: dotdict = dotdict(type=PositionalEncodingEmbedder.__name__, multires=10),
                 t_embedder_cfg: dotdict = dotdict(type=LatentCodeEmbedder.__name__),
                 ):
        super().__init__()

        self.xyz_embedder: PositionalEncodingEmbedder = EMBEDDERS.build(xyz_embedder_cfg)
        self.t_embedder: LatentCodeEmbedder = EMBEDDERS.build(t_embedder_cfg)

        self.xyz_out_dim = self.xyz_embedder.out_dim
        self.t_out_dim = self.t_embedder.out_dim
        self.out_dim = self.xyz_embedder.out_dim + self.t_embedder.out_dim

    def forward(self, xyz: torch.Tensor, t: torch.Tensor, batch: dotdict = None):
        # NOTE: we prefer to store xyz and t separatedly, because those dimensions are essentially different from each other
        # this xyzt is for consistency of the embedder api
        xyz_feat = self.xyz_embedder(xyz, batch)
        t_feat = self.t_embedder(t, batch)
        feat = torch.cat([xyz_feat, t_feat], dim=-1)
        return feat
