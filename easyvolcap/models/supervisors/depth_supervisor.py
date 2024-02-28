# Depth loss module
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg

from easyvolcap.utils.console_utils import *
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor
from easyvolcap.utils.loss_utils import smoothl1, l1, l2, DptLossType, ScaleAndShiftInvariantMSELoss, ScaleAndShiftInvariantMAELoss, ScaleInvariantLogLoss


@SUPERVISORS.register_module()
class DepthSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 dpt_loss_weight: float = 0.0,  # depth supervison
                 dpt_loss_type: DptLossType = DptLossType.SMOOTHL1.name,  # depth loss type
                 scale_invariant_loss_cfg: dotdict = dotdict(),  # scale invariant depth loss
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)
        self.dpt_loss_weight = dpt_loss_weight
        self.dpt_loss_type = DptLossType[dpt_loss_type]
        self.scale_invariant_loss_cfg = scale_invariant_loss_cfg

    @property
    def ssimse_loss(self):
        return self.ssimse_loss_reference[0]

    @property
    def ssimae_loss(self):
        return self.ssimae_loss_reference[0]

    @property
    def silog_loss(self):
        return self.silog_loss_reference[0]

    def compute_depth_loss(self, dpt_map: torch.Tensor, dpt_gt: torch.Tensor, mask: torch.Tensor,
                           H: int, W: int, type=DptLossType.SMOOTHL1):
        if type == DptLossType.SSIMSE:
            if not hasattr(self, 'ssimse_loss_reference'):
                self.ssimse_loss_reference = [ScaleAndShiftInvariantMSELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.ssimse_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SSIMAE:
            if not hasattr(self, 'ssimae_loss_reference'):
                self.ssimae_loss_reference = [ScaleAndShiftInvariantMAELoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.ssimae_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SILOG:
            if not hasattr(self, 'silog_loss_reference'):
                self.silog_loss_reference = [ScaleInvariantLogLoss(**self.scale_invariant_loss_cfg).cuda().to(self.dtype)]
            dpt_loss = self.silog_loss(dpt_map, dpt_gt, mask)
        elif type == DptLossType.SMOOTHL1: dpt_loss = smoothl1(dpt_map[mask], dpt_gt[mask])
        elif type == DptLossType.L1: dpt_loss = l1(dpt_map[mask], dpt_gt[mask])
        elif type == DptLossType.L2: dpt_loss = l2(dpt_map[mask], dpt_gt[mask])
        # TODO: Implement depth continuity loss (scale-invariant)
        # TODO: Implement depth ranking loss (scale-invariant)

        return dpt_loss

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here
        def compute_depth_loss(dpt_map: torch.Tensor, dpt_gt: torch.Tensor, mask: torch.Tensor,
                               H: int = batch.meta.H[0].item(), W: int = batch.meta.W[0].item(),
                               type=self.dpt_loss_type):
            return self.compute_depth_loss(dpt_map, dpt_gt, mask, H, W, type)

        if 'dpt_map' in output and 'dpt' in batch and \
           self.dpt_loss_weight > 0:
            mask = batch.dpt != 0
            dpt_loss = compute_depth_loss(output.dpt_map, batch.dpt, mask)
            scalar_stats.dpt_loss = dpt_loss
            loss += self.dpt_loss_weight * dpt_loss

        return loss
