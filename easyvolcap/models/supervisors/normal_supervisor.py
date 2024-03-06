import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import l1, cos
from easyvolcap.utils.math_utils import normalize


@SUPERVISORS.register_module()
class NormalSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 norm_loss_weight: float = 0.0,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.norm_loss_weight = norm_loss_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'norm_map' in output and 'norm' in batch and self.norm_loss_weight > 0:
            # Transform the normal map to the local coordinate system
            norm_map = normalize(output.norm_map)
            norm_map = norm_map @ batch.R.mT  # convert to view space
            norm_map = normalize(norm_map)

            # Process the ground truth normal map
            norm = batch.norm * 2. - 1.  # this is generally how normals are stored on disk
            norm = normalize(norm)

            # Compute normal loss
            mask = batch.msk[..., 0] > 0.5
            norm_loss = l1(norm_map[mask], norm[mask])  # MARK: SYNC
            norm_loss += cos(norm_map[mask], norm[mask])  # MARK: SYNC

            scalar_stats.norm_loss = norm_loss
            loss += self.norm_loss_weight * norm_loss

        return loss
