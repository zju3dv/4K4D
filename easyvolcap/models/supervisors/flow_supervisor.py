import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.console_utils import dotdict
from easyvolcap.utils.loss_utils import ImgLossType, mse, mIoU_loss
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

@SUPERVISORS.register_module()
class FlowSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module, 
                 flow_loss_weight: float = 0.0,
                 normalize: bool = True,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.flow_loss_weight = flow_loss_weight
        self.normalize = normalize
    
    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'flo_map' in output and 'flow' in batch and 'flow_weight' in batch and \
            self.flow_loss_weight > 0:
            sum_loss = F.l1_loss(output.flo_map, batch.flow, reduction='none').mean(dim=-1, keepdim=True)
            if self.normalize:
                flow_loss = torch.sum((sum_loss * batch.flow_weight) / (torch.sum(batch.flow_weight) + 1e-8))
            else:
                flow_loss = torch.mean((sum_loss * batch.flow_weight))
            scalar_stats.flow_loss = flow_loss
            loss += self.flow_loss_weight * flow_loss
        
        return loss