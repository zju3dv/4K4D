# Default loss module (called supervisor)
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import compute_planes_tv, compute_time_planes_smooth
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class TemporalSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 tv_loss_weight: float = 0.0,  # total variation loss for kplanes
                 time_smooth_weight: float = 0.0,  # time smooth weight
                 time_smooth_prop_weight: float = 0.0,  # time smooth propasal weight

                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.tv_loss_weight = tv_loss_weight
        self.time_smooth_weight = time_smooth_weight
        self.time_smooth_prop_weight = time_smooth_prop_weight

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here

        if self.tv_loss_weight > 0:
            tv_loss = 0
            for net in self.network.networks:
                tv_loss += compute_planes_tv(net.xyzt_embedder.spatial_embedding)
                # if net.xyzt_embedder.temporal_embedding[0].shape[-1]>3: # only add tv loss for temporal embedding if there are more than 3 planes
                # tv_loss += compute_planes_tv(net.xyzt_embedder.temporal_embedding)
            scalar_stats.tv_loss = tv_loss
            loss += self.tv_loss_weight * tv_loss

        if self.time_smooth_weight > 0:
            net = self.network.networks[-1]
            time_smooth_loss = compute_time_planes_smooth(net.xyzt_embedder.temporal_embedding)
            scalar_stats.time_smooth_loss = time_smooth_loss
            loss += self.time_smooth_weight * time_smooth_loss

        if self.time_smooth_prop_weight > 0:
            time_smooth_loss = 0
            for net in self.network.networks[:-1]:
                time_smooth_loss = compute_time_planes_smooth(net.xyzt_embedder.temporal_embedding)
            scalar_stats.time_smooth_prop_loss = time_smooth_loss
            loss += self.time_smooth_prop_weight * time_smooth_loss

        return loss
