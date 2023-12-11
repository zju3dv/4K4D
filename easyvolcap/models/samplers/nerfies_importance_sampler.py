import torch
import numpy as np
from torch import nn
from typing import List
from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.engine import call_from_cfg
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import z_vals_to_s_vals, volume_rendering, ray2xyz, max_dilate_weights, anneal_weights

# TODO: figure out a way to deal with this clumsy import from repeated file names
# while require not extra work from the user to make the import visible
# NOTE: for configuration based initialization, need register
# but for simple building, plain initialization should be fine (rare)
from easyvolcap.models.samplers.uniform_sampler import UniformSampler
from easyvolcap.models.networks.sharable_multilevel_network import SharableMultilevelNetwork
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork


@SAMPLERS.register_module()
class NerfiesImportanceSampler(UniformSampler):
    def __init__(self,
                 network: SharableMultilevelNetwork,
                 n_samples: List[int] = [128, 128],  # number of samples for coarse and fine level
                 bg_brightness: float = -1.0,
                 # Please be aware of argument falling through
                 # We use a recursive module building system, i.e. ImportanceSampler will first build UniformSampler
                 # If you need to enable uniform disparity sampling, you can simply set it here as an argument of the ImportanceSampler
                 # Then it will be passed through to the UniformSampler through kwargs
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # no super init
        self.n_samples = n_samples
        self.bg_brightness = bg_brightness

    def sample_depth(self,
                     ray_o: torch.Tensor, ray_d: torch.Tensor,
                     near: torch.Tensor, far: torch.Tensor,
                     t: torch.Tensor,
                     batch: dotdict,
                     ):
        # ray_o: B, P, 3
        # ray_d: B, P, 3
        # t: B, P, 1

        # NOTE: principles for trailing dimensions
        # 1. if it's a split dimension: rgba, the a, or xyzt: the t, retain the last 1 dim
        # 2. if it's not to be merged (nor was it from another dimension), remove it, like depth of a sample: NOTE: if we're talking about depth of a pixel, retain it

        # Store stuff for loss
        output = dotdict()
        # For supervision of coarse network
        output.rgb_maps_prop = []
        output.weights_prop = []
        output.jacobian_prop = []
        output.bg_colors_prop = []

        for round in range(len(self.n_samples)):
            n_samples = self.n_samples[round]  # number of samples for this round

            # Computing z_vals (return) and s_vals
            # Perform sampling (uniform or weighted)
            if round == 0:
                # The first round is always uniform (in whatever space defined)
                z_vals = super().sample_depth(ray_o, ray_d, near, far, t, batch, n_samples=n_samples)

            # Compute perform importance sampling on other rounds
            else:  # round != 0
                s_vals = s_vals.detach()  # avoid back propagation through sampling (as in mipnerf360)
                weights = weights.detach()  # avoid back propagation through sampling

                # On consecutive rounds, we need to recompute the weights
                # according to mipnerf360
                z_vals = super().sample_depth(ray_o, ray_d, near, far, t, batch, n_samples=n_samples, s_vals=s_vals, weights=weights)

            # From near, far to 0, 1
            # MARK: exposed string key
            s_vals = z_vals_to_s_vals(z_vals, near, far, g=self.g)  # no space conversion

            # Compute weights (rgb, occ, volume rendering)
            # Perform volume rendering on last but not least round
            if round != len(self.n_samples) - 1:
                xyz_c, dir_c, t_c, dist_c = ray2xyz(ray_o, ray_d, t, z_vals)
                # perform forward pass to get rgb and occ
                rgb, occ = self.network.compute_coarse(VolumetricVideoNetwork.forward.__name__, round, xyz_c, dir_c, t_c, dist_c, batch)

                # Nasty shapes
                sh = z_vals.shape  # B, P, S
                occ = occ.view(sh + (-1,))  # B, P, S, 1
                rgb = rgb.view(sh + (-1,))  # B, P, S, 3

                # TODO: PERF
                if self.bg_brightness < 0 and self.training: bg_color = torch.rand_like(rgb[..., 0, :])  # remove sample dim (in 0 -> 1 already)
                else: bg_color = torch.full_like(rgb[..., 0, :], self.bg_brightness)  # remove sample dim
                weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_color)  # B, P, S

                # Store stuff in the batch variable (the field should be reserved)
                output.rgb_maps_prop.append(rgb_map)
                output.bg_colors_prop.append(bg_color)
                output.weights_prop.append(weights)
                if 'jacobian' in batch.output.keys(): output.jacobian_prop.append(batch.output.jacobian)  # val network do not compute jacobian

        # Get the importance weights
        # Construct cdf
        # Perform inverse cdf sampling

        output.s_vals = s_vals  # for rendering depth and stuff? (should retain gradients)
        output.z_vals = z_vals  # for rendering depth and stuff? (should retain gradients)
        batch.output.update(output)
        return z_vals.detach()  # is this finally the problem? no propagation through sampling?
