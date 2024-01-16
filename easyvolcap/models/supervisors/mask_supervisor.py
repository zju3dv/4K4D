# Default loss module (called supervisor)
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import mse, mIoU_loss
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class MaskSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 msk_loss_weight: float = 0.0,  # mask mIoU loss
                 msk_mse_weight: float = 0.0,  # mask mse weight
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.msk_loss_weight = msk_loss_weight
        self.msk_mse_weight = msk_mse_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here
        if 'acc_map' in output and 'msk' in batch and \
           self.msk_loss_weight > 0:
            msk_loss = mIoU_loss(output.acc_map, batch.msk)
            scalar_stats.msk_loss = msk_loss
            loss += self.msk_loss_weight * msk_loss

        if 'acc_map' in output and 'msk' in batch and \
           self.msk_mse_weight > 0:
            msk_loss = mse(output.acc_map, batch.msk)
            scalar_stats.msk_mse = msk_loss
            loss += self.msk_mse_weight * msk_loss

        return loss
