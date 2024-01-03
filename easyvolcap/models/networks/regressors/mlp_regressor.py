
import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import MLP, get_function
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.networks.regressors.tcnn_mlp_regressor import TcnnMlpRegressor


@REGRESSORS.register_module()
class MlpRegressor(nn.Module):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 3,
                 width: int = 256,
                 depth: int = 8,
                 actvn: nn.Module = nn.ReLU(),
                 out_actvn: nn.Module = nn.Identity(),

                 backend: Literal['tcnn', 'torch'] = 'torch',
                 otype: str = 'FullyFusedMLP',  # might lead to performance degredation, only used with backend == tcnn
                 dtype: str = 'float',
                 ):
        # Simply an MLP wrapper
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.backend = backend
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        actvn = get_function(actvn) if isinstance(actvn, str) else actvn
        out_actvn = get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn

        if backend == 'torch':
            self.mlp = MLP(in_dim, width, depth, out_dim, actvn=actvn, out_actvn=out_actvn, dtype=dtype)
            self.mlp.linears[-1].bias.data.normal_(0, std=1e-4)  # small displacement by default
        elif backend == 'tcnn':
            if depth <= 1:
                log(yellow(f'`tcnn` requires at least 1 hidden layer, depth {depth} -> {3}'))
                depth = 2
            self.mlp = TcnnMlpRegressor(in_dim, out_dim, width, depth, out_actvn='None', otype=otype, dtype=dtype)
            self.out_actvn = out_actvn

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        if self.backend == 'torch':
            return self.mlp(feat)
        else:
            return self.out_actvn(self.mlp(feat))
