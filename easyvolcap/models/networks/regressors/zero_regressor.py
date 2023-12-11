import torch
from torch import nn
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict


@REGRESSORS.register_module()
class ZeroRegressor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        return torch.zeros(feat.shape[:-1] + (self.out_dim,), device=feat.device, dtype=feat.dtype)
