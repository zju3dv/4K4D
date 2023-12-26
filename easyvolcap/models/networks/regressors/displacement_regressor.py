
import torch
from torch import nn
from typing import Literal

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.net_utils import MLP
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor


@REGRESSORS.register_module()
class DisplacementRegressor(nn.Module):
    # Outputs displacement (used as residual displacement in dnerf deformation)
    def __init__(self,
                 in_dim: int,
                 scale: float = 0.15,
                 zero_canonical: bool = False,
                 **kwargs,
                 ):
        # Simply an MLP wrapper
        super().__init__()

        self.in_dim = in_dim
        self.scale = scale
        self.zero_canonical = zero_canonical
        self.mlp: MlpRegressor = MlpRegressor(in_dim, **kwargs)  # wtf?
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def _pre_load_state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # TODO: Something is not quite right here
        # Processs legacy models
        if f'{prefix}mlp.linears.0.weight' in state_dict:
            for i in range(len(self.mlp.mlp.linears)):
                state_dict[f'{prefix}mlp.mlp.linears.{i}.weight'] = state_dict[f'{prefix}mlp.linears.{i}.weight'].clone()
                state_dict[f'{prefix}mlp.mlp.linears.{i}.bias'] = state_dict[f'{prefix}mlp.linears.{i}.bias'].clone()
                del state_dict[f'{prefix}mlp.linears.{i}.weight']
                del state_dict[f'{prefix}mlp.linears.{i}.bias']

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        dxyz = self.mlp(feat) * self.scale
        if self.zero_canonical: dxyz[torch.where(batch.t == 0.0)] = 0
        return dxyz
