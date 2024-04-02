# This file contains optimizable camera parameters
# Implemented in SO3xR3, exponential map of rotation and translation from screw rt motion

import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join

from easyvolcap.engine import cfg
from easyvolcap.engine import CAMERAS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import NoopModule


@CAMERAS.register_module()
class NoopCamera(NoopModule):  # TODO: Implement intrinsics optimization
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward_srcs(self, batch: dotdict):
        return batch

    def forward_cams(self, batch: dotdict):
        return batch

    def forward_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, batch, use_z_depth: bool = False, correct_pix: bool = True):
        return ray_o, ray_d

    def forward(self, ray_o: torch.Tensor, ray_d: torch.Tensor, batch, use_z_depth: bool = False, correct_pix: bool = True):
        return ray_o, ray_d, batch
