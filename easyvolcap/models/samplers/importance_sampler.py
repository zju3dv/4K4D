import torch
import numpy as np
from torch import nn
from typing import List
from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.engine import call_from_cfg
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.nerf_utils import render_weights, ray2xyz, linear_sampling
from easyvolcap.utils.prop_utils import z_vals_to_s_vals, max_dilate_weights, anneal_weights, importance_sampling, s_vals_to_z_vals

# TODO: figure out a way to deal with this clumsy import from repeated file names
# while require not extra work from the user to make the import visible
# NOTE: for configuration based initialization, need register
# but for simple building, plain initialization should be fine (rare)
from easyvolcap.models.samplers.uniform_sampler import UniformSampler
from easyvolcap.models.networks.multilevel_network import MultilevelNetwork
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork


@SAMPLERS.register_module()
class ImportanceSampler(UniformSampler):
    def __init__(self,
                 network: MultilevelNetwork,
                 n_samples: List[int] = [64, 64, 64],  # number of samples per-round
                 fine_freespace_ratio: float = 0.25,  # number of random samples # FIXME: DO NOT SET TO ABOVE ZERO, not implemented to insert into samples

                 normalize_weight: bool = True,  # these will lead to inconvergence...
                 dilate_weight: bool = False,  # these will lead to inconvergence...
                 anneal_weight: bool = False,  # these will lead to inconvergence...
                 anneal_slope: float = 10.0,
                 anneal_max_steps: float = 1000,
                 dilation_bias: float = 0.0025,
                 dilation_multiplier: float = 0.5,

                 # Please be aware of argument falling through
                 # We use a recursive module building system, i.e. ImportanceSampler will first build UniformSampler
                 # If you need to enable uniform disparity sampling, you can simply set it here as an argument of the ImportanceSampler
                 # Then it will be passed through to the UniformSampler through kwargs
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # no super init
        self.n_samples = n_samples
        self.n_freespace = int(n_samples[-1] * fine_freespace_ratio)
        self.n_samples[-1] -= self.n_freespace  # last level performs some random sampling too

        self.normalize_weight = normalize_weight
        self.dilate_weight = dilate_weight
        self.dilation_bias = dilation_bias
        self.dilation_muliplier = dilation_multiplier
        self.anneal_weight = anneal_weight
        self.anneal_slope = anneal_slope
        self.anneal_max_steps = anneal_max_steps

        # a little bit awkward looking but it works
        # NOTE: manually control uniform_sampler's count
        # self.uniform_sampler: UniformSampler = SAMPLERS.build(uniform_sampler_cfg, network=network)

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
        B, P, _ = near.shape

        # Store stuff for loss
        output = dotdict()
        output.s_vals_prop = []
        output.weights_prop = []
        for round in range(len(self.n_samples)):
            n_samples = self.n_samples[round]  # number of samples for this round

            # Computing z_vals (return) and s_vals
            # Perform sampling (uniform or weighted)
            if round == 0:

                # The first round is always uniform (in whatever space defined)
                s_vals = linear_sampling(B, P, n_samples, device=ray_o.device, dtype=ray_o.dtype, perturb=self.training)  # 0 -> B, P, S
                z_vals = s_vals_to_z_vals(s_vals, near, far, g=self.g, ig=self.ig)

            # Compute perform importance sampling on other rounds
            else:  # round != 0

                s_vals = s_vals.detach()  # avoid back propagation through sampling (as in mipnerf360)
                weights = weights.detach()  # avoid back propagation through sampling

                # Maybe perform weight dilation
                if self.dilate_weight:
                    dilation = self.dilation_bias + self.dilation_muliplier / np.prod(self.n_samples[:round + 1])
                    s_vals, weights = max_dilate_weights(s_vals, weights, dilation, domain=(0., 1.), renormalize=True)
                    s_vals = s_vals[..., 1:-1]
                    weights = weights[..., 1:-1]

                # Maybe perform weight annealing
                if self.anneal_weight:
                    iter = batch.meta.iter
                    frac = (iter / self.anneal_max_steps)
                    weights = anneal_weights(s_vals, weights, frac, self.anneal_slope)  # anneal the weights as a function of iter

                # On consecutive rounds, we need to recompute the weights
                # according to mipnerf360
                s_vals = importance_sampling(s_vals, weights, n_samples, perturb=self.training)  # 0 -> B, P, S,
                z_vals = s_vals_to_z_vals(s_vals, near, far, g=self.g, ig=self.ig)

            # Compute weights (rgb, occ, volume rendering)
            # Perform volume rendering on last but not least round
            if round != len(self.n_samples) - 1:
                xyz_c, dir_c, t_c, dist_c = ray2xyz(ray_o, ray_d, t, z_vals)
                occ = self.network.compute_coarse(VolumetricVideoNetwork.occ.__name__, round, xyz_c, t_c, dist_c, batch)

                # Nasty shapes
                sh = z_vals.shape  # B, P, S
                occ = occ.view(sh + (-1,))  # B, P, S, 1
                weights = render_weights(occ)  # B, P, S

                if self.normalize_weight:
                    weights = weights / (weights.sum(dim=-1, keepdim=True) + torch.finfo(torch.float32).eps)

                # Store stuff in the batch variable (the field should be reserved)
                output.s_vals_prop.append(s_vals)
                output.weights_prop.append(weights)

        # Get the importance weights
        # Construct cdf
        # Perform inverse cdf sampling

        # Random sample along the ray to reduce empty space artifaces
        if self.n_freespace > 0 and self.training:  # only add in these perturbations during training
            s_vals_rand = linear_sampling(B, P, self.n_freespace, device=ray_o.device, dtype=ray_o.dtype, perturb=self.training)  # 0 -> B, P, S
            z_vals_rand = s_vals_to_z_vals(s_vals_rand, near, far, g=self.g, ig=self.ig)
            z_vals = torch.cat([z_vals, z_vals_rand], dim=-1).sort(dim=-1)[0]  # near to far
            s_vals = torch.cat([s_vals, s_vals_rand], dim=-1).sort(dim=-1)[0]  # near to far

        # np.testing.assert_array_less(s_vals[..., :-1].detach().cpu().numpy(), s_vals[..., 1:].detach().cpu().numpy())

        output.s_vals = s_vals  # for rendering depth and stuff? (should retain gradients)
        output.z_vals = z_vals  # for rendering depth and stuff? (should retain gradients)
        batch.output.update(output)
        return z_vals.detach()  # is this finally the problem? no propagation through sampling?
