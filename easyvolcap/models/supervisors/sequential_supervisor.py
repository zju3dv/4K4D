# The user might specify multiple supervisors
# Call the supervise function, aggregate losses and stats
# Default loss module (called supervisor)
import copy
import torch
import numpy as np
from torch import nn
from typing import Union, List

from easyvolcap.engine import SUPERVISORS
from easyvolcap.utils.console_utils import *
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.mask_supervisor import MaskSupervisor
from easyvolcap.models.supervisors.flow_supervisor import FlowSupervisor
from easyvolcap.models.supervisors.depth_supervisor import DepthSupervisor
from easyvolcap.models.supervisors.normal_supervisor import NormalSupervisor
from easyvolcap.models.supervisors.opacity_supervisor import OpacitySupervisor
from easyvolcap.models.supervisors.proposal_supervisor import ProposalSupervisor
from easyvolcap.models.supervisors.geometry_supervisor import GeometrySupervisor
from easyvolcap.models.supervisors.temporal_supervisor import TemporalSupervisor
from easyvolcap.models.supervisors.displacement_supervisor import DisplacementSupervisor
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor
from easyvolcap.models.supervisors.motion_consistency_supervisor import MotionConsistencySupervisor


@SUPERVISORS.register_module()
class SequentialSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,
                 supervisor_cfgs: List[dotdict] = [
                     dotdict(type=MaskSupervisor.__name__),
                     dotdict(type=FlowSupervisor.__name__),
                     dotdict(type=DepthSupervisor.__name__),
                     dotdict(type=MotionConsistencySupervisor.__name__),
                     dotdict(type=NormalSupervisor.__name__),
                     dotdict(type=OpacitySupervisor.__name__),
                     dotdict(type=ProposalSupervisor.__name__),
                     dotdict(type=GeometrySupervisor.__name__),
                     dotdict(type=TemporalSupervisor.__name__),
                     dotdict(type=DisplacementSupervisor.__name__),
                     dotdict(type=VolumetricVideoSupervisor.__name__), # NOTE: Put this last for PSNR to be displayed last along with full loss
                 ],
                 **kwargs,
                 ):
        kwargs = dotdict(kwargs)  # for recursive update
        call_from_cfg(super().__init__, kwargs, network=network)
        supervisor_cfgs = [copy.deepcopy(kwargs).update(supervisor_cfg) for supervisor_cfg in supervisor_cfgs]  # for recursive update
        self.supervisors: nn.ModuleList[VolumetricVideoSupervisor] = nn.ModuleList([SUPERVISORS.build(supervisor_cfg, network=network) for supervisor_cfg in supervisor_cfgs])

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        for supervisor in self.supervisors:
            loss = supervisor.compute_loss(output, batch, loss, scalar_stats, image_stats)  # loss will be added
        return loss
