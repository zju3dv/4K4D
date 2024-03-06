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
class OpacitySupervisor(VolumetricVideoSupervisor):
    def __init__(self, 
                 network: nn.Module, 
                 ent_loss_weight: float = 0.0, 
                 **kwargs):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.ent_loss_weight = ent_loss_weight
    
    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'occ' in output and self.ent_loss_weight > 0:
            ent_loss = -torch.mean(output.occ * torch.log(output.occ))
            scalar_stats.ent_loss = ent_loss
            loss += self.ent_loss_weight * ent_loss
        
        return loss