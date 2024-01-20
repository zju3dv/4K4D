import torch
import numpy as np

from torch import nn
from sympy import nextprime
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import make_buffer, make_params
from easyvolcap.engine import EMBEDDERS
from easyvolcap.engine import cfg
from typing import List


@EMBEDDERS.register_module()
class TcnnHashEmbedder(nn.Module):
    from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera

    def __init__(self,
                 n_levels=16,
                 n_features_per_level=2,
                 b=1.38,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 interpolation='Linear',

                 bounds: List[List[int]] = OptimizableCamera.square_bounds,  # MARK: GLOBAL
                 in_dim: int = 3,
                 predefined_sizes: List[int] = None,
                 dtype=torch.float,  # MARK: Using float as default (but tcnn performs the best when using half)

                 make_mask: bool = False,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.t = log2_hashmap_size
        self.n_levels = n_levels
        self.n_entries_per_level = nextprime(2**log2_hashmap_size)
        self.interpolation = interpolation

        # if predefined_sizes is not None:
        #     log(yellow(f'non-square dims {cyan(predefined_sizes)} -> extra memory & storage'))

        self.b = b
        self.f = n_features_per_level
        self.base_resolution = base_resolution
        self.bounds = make_buffer(torch.as_tensor(bounds))  # 2, 3

        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        encoding_config = dotdict()
        encoding_config.otype = 'HashGrid'
        encoding_config.n_levels = n_levels
        encoding_config.n_features_per_level = n_features_per_level
        encoding_config.log2_hashmap_size = log2_hashmap_size
        encoding_config.base_resolution = base_resolution
        encoding_config.per_level_scale = b  # per-level growth factor
        encoding_config.interpolation = interpolation

        import tinycudann as tcnn  # lasy imports
        try:
            self.tcnn_encoding = tcnn.Encoding(
                dtype=dtype,
                n_input_dims=in_dim,
                encoding_config=encoding_config,
            )
        except RuntimeError as e:
            retry = min(8, 2 ** int(np.log2(n_features_per_level)))
            log(red(f'`tcnn` backbone only supports n_features_per_level 1, 2, 4, 8, current: {n_features_per_level}, will retry with {retry}'))
            self.tcnn_encoding = tcnn.Encoding(
                dtype=dtype,
                n_input_dims=in_dim,
                encoding_config={**encoding_config, 'n_features_per_level': retry},
            )

        self.in_dim = in_dim
        self.out_dim = self.tcnn_encoding.n_output_dims
        self.make_mask = make_mask
        if self.make_mask: self.hash_encoding_mask = make_buffer(torch.ones(n_levels * n_features_per_level, dtype=dtype))

    def forward(self, xyz: torch.Tensor, batch: dotdict = None):
        bash = xyz.shape  # batch shape
        xyz = xyz.view(-1, xyz.shape[-1])
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3
        # log(xyz.min(-2)[0], xyz.max(-2)[0])
        # https://github.com/NVlabs/tiny-cuda-nn/issues/286
        # The meat of the operation
        val: torch.Tensor = self.tcnn_encoding(xyz)
        val = val.view(*bash[:-1], val.shape[-1])

        if self.make_mask: return val * self.hash_encoding_mask
        else: return val

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.f:] = 0.0
