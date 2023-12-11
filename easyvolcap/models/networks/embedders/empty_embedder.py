# Literally return the input as is
import torch
from torch import nn
from easyvolcap.engine import EMBEDDERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import DoesNotCareAboutStateDict

@EMBEDDERS.register_module()
class EmptyEmbedder(DoesNotCareAboutStateDict):
    def __init__(self, out_dim=0, **kwargs):
        super().__init__()
        self.out_dim = 0  # no embedding, no output

    def forward(self, inputs: torch.Tensor, batch: dotdict = None):
        return torch.zeros(*inputs.shape[:-1], 0, device=inputs.device, dtype=inputs.dtype)  # empty tensor
