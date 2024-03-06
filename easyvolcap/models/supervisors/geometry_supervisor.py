# Default loss module (called supervisor)
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import eikonal, lossfun_zip_outer, reg_raw_crit, compute_val_pair_around_range
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor


@SUPERVISORS.register_module()
class GeometrySupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,

                 # SDF model
                 eikonal_loss_weight: float = 0.0,
                 zip_prop_loss_weight: float = 0.0,
                 curvature_loss_weight: float = 0.0,

                 # Jacobian / normal consistency loss
                 norm_smooth_weight: float = 0.0,
                 norm_smooth_range: float = 0.025,  # smaller smoothing range for a more consistent training
                 surf_occ_min: float = 0.2,
                 surf_occ_max: float = 0.8,
                 norm_smooth_pts_start: int = 1024,  # start computing this loss only after this number of points has been reached
                 #  norm_smooth_ann_iter: int = 10 * 500,
                 norm_smooth_ann_iter: int = 1,

                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        self.zip_prop_loss_weight = zip_prop_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.curvature_loss_weight = curvature_loss_weight
        self.norm_smooth_pts_start = norm_smooth_pts_start
        self.norm_smooth_ann_iter = norm_smooth_ann_iter
        self.norm_smooth_weight = norm_smooth_weight
        self.norm_smooth_range = norm_smooth_range
        self.surf_occ_min = surf_occ_min
        self.surf_occ_max = surf_occ_max

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        # Compute the actual loss here
        if self.norm_smooth_weight > 0.0:
            # Find points close to the surface, by thresholding the occupancy value (0.2-0.8)
            # Sample new points around them by adding a small purturbing vector (0.025 2.5cm)
            # Compute their respective gradient vector (call the network's respective APIs)
            # Regularize the difference in their jacobian vector
            mask = (output.occ[..., 0] > self.surf_occ_min) & (output.occ[..., 0] < self.surf_occ_max)  # valid point mask
            if mask.sum() > 0:
                xyz = output.xyz[mask][None]  # MARK: SYNC, 1, N, 3
                time = batch.t[:, None, None].expand(output.xyz[..., :1].shape).to(output.xyz.dtype)
                time = time[mask][None]   # MARK: SYNC, 1, N, 1

                neighbor = xyz + (torch.rand_like(xyz) - 0.5) * self.norm_smooth_range
                xyz = torch.cat([xyz, neighbor], dim=-2)  # cat in n_masked dim
                time = time.repeat(1, 2, 1)

                norm_smooth = 0
                weight = 0
                for network in self.network.networks[-1:]:  # only last level
                    diff = network.gradient(xyz, time, batch)  # (n_batch, n_masked, 3)
                    norm, weight = reg_raw_crit(diff, batch.meta.iter.item(), self.norm_smooth_weight, self.norm_smooth_ann_iter)
                    norm_smooth += norm

                scalar_stats.norm_smooth = norm_smooth
                loss += weight * norm_smooth

        if self.eikonal_loss_weight > 0.0 and 'gradients' in output:
            gradients = output.gradients
            eikonal_loss = eikonal(gradients)
            scalar_stats.eikonal_loss = eikonal_loss
            loss += self.eikonal_loss_weight * eikonal_loss

        if self.curvature_loss_weight > 0.0 and 'sampled_sdf' in output:
            delta = self.network.finite_diff_delta
            centered_sdf = output.sdf
            sourounding_sdf = output.sampled_sdf
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta ** 2)
            curvature_loss = curvature.abs().mean() * self.network.curvature_loss_multi_factor
            scalar_stats.curvature_loss = curvature_loss
            loss += self.curvature_loss_weight * curvature_loss

        if 's_vals_prop' in output and 'weights_prop' in output and \
           len(output.s_vals_prop) and len(output.weights_prop) and \
           's_vals' in output and 'weights' in output and \
           self.zip_prop_loss_weight > 0:
            zip_prop_loss = 0
            pulse_width = [0.03, 0.003, 0.0003]
            for i in range(len(output.s_vals_prop)):
                zip_prop_loss += lossfun_zip_outer(
                    output.s_vals.detach(), output.weights.detach(),
                    output.s_vals_prop[i], output.weights_prop[i],
                    pulse_width=pulse_width[i])
            scalar_stats.zip_prop_loss = zip_prop_loss
            loss += self.zip_prop_loss_weight * zip_prop_loss

        return loss
