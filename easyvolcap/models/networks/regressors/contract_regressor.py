# mipnerf360 space contraction
import torch
import numpy as np
from torch import nn
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.bound_utils import contract, get_bounds
from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera


@REGRESSORS.register_module()
class ContractRegressor(nn.Module):
    radius = (OptimizableCamera.bounds[1] - OptimizableCamera.bounds[0]).max()  # make it easier on object-centric datasets

    def __init__(self,
                 in_dim: int = 3,
                 radius: float = radius,  # -> 10.0m?, bad convergence if radius too small
                 p: float = torch.inf,
                 normalize: bool = False,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.radius = radius
        self.p = p
        self.normalize = normalize

    def forward(self, xyz: torch.Tensor, batch: dotdict = None):
        xyz = contract(xyz, self.radius, self.p)
        if self.normalize:
            xyz = xyz / self.radius
        return xyz
