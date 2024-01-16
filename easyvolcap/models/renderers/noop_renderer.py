import torch
from torch import nn
from typing import List
from easyvolcap.engine import RENDERERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import VolumetricVideoModule


@RENDERERS.register_module()
class NoopRenderer(VolumetricVideoModule):  # should not contain optimizables
    def __init__(self,
                 network: nn.Module,
                 **kwargs,  # ignore other arguments
                 ):
        super().__init__(network)
        self.forward = self.render

    def render(self, rgb: torch.Tensor, occ: torch.Tensor, batch: dotdict):
        # raw: main renderable data
        # batch: other useful resources
        return None
