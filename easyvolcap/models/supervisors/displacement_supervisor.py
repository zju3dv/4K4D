# Default loss module (called supervisor)
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import elastic_crit, l2_reg, ElasticLossReduceType
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class DisplacementSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,

                 resd_loss_weight: float = 0.0,  # make resd smaller
                 elas_loss_weight: float = 0.0,
                 elas_reduce_type: ElasticLossReduceType = ElasticLossReduceType.WEIGHT.name,

                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.resd_loss_weight = resd_loss_weight
        self.elas_loss_weight = elas_loss_weight
        self.elas_reduce_type = ElasticLossReduceType[elas_reduce_type]

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here

        if 'jacobian_prop' in output and 'weights_prop' in output and \
           len(output.jacobian_prop) and len(output.weights_prop) and \
           self.elas_loss_weight > 0:
            prop_elas_loss = 0
            for i in range(len(output.jacobian_prop)):
                jacobian = output.jacobian_prop[i]
                weights = output.weights_prop[i]
                prop_elas_loss_i = elastic_crit(jacobian)  # (B, N)
                if self.elas_reduce_type == ElasticLossReduceType.WEIGHT:
                    prop_elas_loss_i = prop_elas_loss_i * weights.reshape(output.weights.shape[0], -1)
                prop_elas_loss_i = torch.sum(prop_elas_loss_i, dim=-1).mean()
                prop_elas_loss += prop_elas_loss_i
            scalar_stats.prop_elas_loss = prop_elas_loss
            loss += self.elas_loss_weight * prop_elas_loss

        if 'resd_prop' in output and len(output.resd_prop) and \
           self.resd_loss_weight > 0:
            prop_resd_loss = 0
            for i in range(len(output.resd_prop)):
                prop_resd_loss_i = (output.resd_prop[i] ** 2).mean()
                prop_resd_loss += prop_resd_loss_i
            scalar_stats.prop_resd_loss = prop_resd_loss
            loss += self.resd_loss_weight * prop_resd_loss

        if 'jacobian' in output and 'weights' in output and 'jacobian_prop' not in output and \
           self.elas_loss_weight > 0:
            elas_loss = elastic_crit(output.jacobian)   # (B, N)
            if self.elas_reduce_type == ElasticLossReduceType.WEIGHT:
                elas_loss = elas_loss * output.weights.reshape(output.weights.shape[0], -1)
            elas_loss = torch.sum(elas_loss, dim=-1).mean()
            scalar_stats.elas_loss = elas_loss
            loss += self.elas_loss_weight * elas_loss

        # residual displacement supervision for dnerf
        if 'resd' in output and 'resd_prop' not in output and \
           self.resd_loss_weight > 0:
            resd_loss = l2_reg(output.resd)
            scalar_stats.resd_loss = resd_loss
            loss += self.resd_loss_weight * resd_loss

        return loss
