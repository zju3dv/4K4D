import torch
from easyvolcap.engine import MODELS
from easyvolcap.utils.net_utils import DoesNotCareAboutStateDict, make_buffer


@MODELS.register_module()
class NoopModel(DoesNotCareAboutStateDict):
    def __init__(self,
                 **kwargs,  # suppress warnings
                 ):
        super().__init__()
        self.device_tracker = make_buffer(torch.empty(0))
