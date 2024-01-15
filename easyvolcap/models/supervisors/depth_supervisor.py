# Default loss module (called supervisor)
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import eikonal, lossfun_zip_outer
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class DepthSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 dpt_loss_weight: float = 0.0,  # depth supervison
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.dpt_loss_weight = dpt_loss_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here

        if 'dpt_map' in output and 'dpt' in batch and \
           self.dpt_loss_weight > 0:
            # TODO: Implement depth loss here (scale-variant)
            # TODO: Implement scale invariant depth loss here
            # TODO: Implement depth continuity loss (scale-invariant)
            # TODO: Implement depth ranking loss (scale-invariant)
            dpt_map = output.dpt_map
            dpt_gt = batch.dpt

            mask = dpt_gt != 0
            dpt_loss = F.smooth_l1_loss(dpt_map[mask], dpt_gt[mask])  # MARK: SYNC

            scalar_stats.dpt_loss = dpt_loss
            loss += self.dpt_loss_weight * dpt_loss

        return loss
