# Literally return the input as is
import torch
from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import NoopModule


@EMBEDDERS.register_module()
class NoopEmbedder(NoopModule):
    def __init__(self,
                 in_dim: int,
                 **kwargs):
        super().__init__()
        # Should this even exist?
        self.in_dim = in_dim
        self.out_dim = in_dim  # no embedding, no output

    def forward(self, inputs: torch.Tensor, batch: dotdict = None):
        return inputs
