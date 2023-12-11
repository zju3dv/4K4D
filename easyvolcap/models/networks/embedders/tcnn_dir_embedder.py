import torch
from torch import nn

from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.net_utils import MLP
from easyvolcap.utils.base_utils import dotdict


@EMBEDDERS.register_module()
class TcnnDirEmbedder(nn.Module):
    def __init__(self,
                 # Should only be used for api consistency
                 in_dim: int = 3,  # only for configurting output shape
                 degree: int = 3,
                 otype: str = 'SphericalHarmonics',
                 dtype: str = torch.float,
                 **kwargs,
                 ):
        super().__init__()
        self.in_dim = in_dim

        encoding_config = dotdict()
        encoding_config.n_dims_to_encode = in_dim
        encoding_config.otype = otype
        encoding_config.degree = degree
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        import tinycudann as tcnn  # lasy imports
        self.enc = tcnn.Encoding(dtype=dtype,
                                 n_input_dims=in_dim,
                                 encoding_config=encoding_config)
        self.out_dim = self.enc.n_output_dims

    def forward(self, input: torch.Tensor, batch: dotdict = None):
        # inputs: B, N, 3
        feat: torch.Tensor = self.enc(input.view(-1, input.shape[-1])).view(*input.shape[:-1], -1)  # no sequential and forward_with_previous split
        return feat
