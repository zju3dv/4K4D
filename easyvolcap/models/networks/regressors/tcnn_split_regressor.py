import torch
from torch import nn
from easyvolcap.engine import REGRESSORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.base_utils import dotdict

from easyvolcap.models.networks.regressors.tcnn_mlp_regressor import TcnnMlpRegressor


@REGRESSORS.register_module()
class TcnnSplitRegressor(TcnnMlpRegressor):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 splits=[1, 256],
                 #  activs=nn.ModuleList([Modulized(raw2alpha), nn.Identity()])
                 #  activs=nn.ModuleList([nn.Sigmoid(), nn.Identity()])
                 #  activs=nn.ModuleList([nn.ReLU(), nn.Identity()])
                 activs=nn.ModuleList([nn.Softplus(), nn.Identity()]),
                 **kwargs,
                 ):
        self.out_dim = sum(splits)
        call_from_cfg(super().__init__, kwargs, out_dim=self.out_dim, out_actvn='None')
        self.splits = splits
        self.activs = activs  # will this register as module?

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        feat: torch.Tensor = self.mlp(feat.view(-1, feat.shape[-1])).view(*feat.shape[:-1], -1)  # no sequential and forward_with_previous split
        feat = feat.split(self.splits, dim=-1)
        feat = [self.activs[i](f) for i, f in enumerate(feat)]  # occ, feature
        feat = torch.cat(feat, dim=-1)
        return feat
