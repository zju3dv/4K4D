import torch
from torch import nn
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import NoopModule


@REGRESSORS.register_module()
class EmptyRegressor(NoopModule):
    def __init__(self, in_dim: int, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        return torch.zeros(*feat.shape[:-1], 0, device=feat.device, dtype=feat.dtype)
