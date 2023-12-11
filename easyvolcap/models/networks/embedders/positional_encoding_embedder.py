import torch
from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import make_buffer


@EMBEDDERS.register_module()
class PositionalEncodingEmbedder(nn.Module):
    def __init__(self,
                 multires,
                 periodic_fns=[torch.sin, torch.cos],
                 retain_input=True,

                 # Should only be used for api consistency
                 in_dim: int = 3,  # only for configurting output shape
                 ):
        super().__init__()
        freq_bands = 2.**torch.linspace(0., multires - 1, steps=multires)  # (multires)
        freq_bands = freq_bands[..., None, None].expand(multires, len(periodic_fns), 1)  # (multires, 2, 1)
        self.freq_bands = make_buffer(freq_bands)
        self.multires = multires
        self.periodic_fns = periodic_fns
        self.retain_input = retain_input
        self.in_dim = in_dim

    def get_dim(self, dim):
        return self.freq_bands.numel() * dim + (dim if self.retain_input else 0)

    @property
    def out_dim(self):
        return self.get_dim(self.in_dim)  # actually not limited to this

    def forward(self, input: torch.Tensor, batch: dotdict = None):
        # inputs: B, N, 3
        sh = input.shape
        feat = input.view(*sh[:-1], 1, 1, sh[-1])  # (B, N, 1, 1, 3)
        feat = feat * self.freq_bands[(None,) * (len(sh) - 1)]  # (B, N, 1, 1, 3) * (1, 1, multires, 2, 3) -> (B, N, multires, 2, 3)
        feat = torch.cat([self.periodic_fns[i](t) for i, t in enumerate(feat.split(1, dim=-2))], dim=-2)
        feat = feat.view(*sh[:-1], self.freq_bands.numel() * sh[-1])  # (B, N, embed_dim - 3?)
        if self.retain_input: feat = torch.cat([input, feat], dim=-1)
        return feat
