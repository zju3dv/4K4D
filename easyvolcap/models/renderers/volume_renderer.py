from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import torch
from torch import nn
from typing import List
from easyvolcap.engine import RENDERERS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import VolumetricVideoModule
from easyvolcap.utils.nerf_utils import volume_rendering
from easyvolcap.utils.prop_utils import weighted_percentile


@RENDERERS.register_module()
class VolumeRenderer(VolumetricVideoModule):  # should not contain optimizables
    def __init__(self,
                 network: nn.Module,
                 bg_brightness: float = -1.0,
                 use_median_depth: bool = True,  # will find median depth instead of performing integration
                 normalize_weights: bool = True,
                 **kwargs,
                 ):
        super().__init__(network)
        # < 0 -> random, 0 -> black, 1 -> white
        self.bg_brightness = bg_brightness
        self.use_median_depth = use_median_depth
        self.normalize_weights = normalize_weights
        self.forward = self.render

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        from imgui_bundle import imgui
        from imgui_bundle import imgui_toggle
        toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2)
        self.use_median_depth = imgui_toggle.toggle('Use median depth', self.use_median_depth, config=toggle_ios_style)[1]
        self.normalize_weights = imgui_toggle.toggle('Normalize weights', self.normalize_weights, config=toggle_ios_style)[1]

    def render(self, rgb: torch.Tensor, occ: torch.Tensor, batch: dotdict):
        # raw: main renderable data
        # batch: other useful resources

        if 'z_vals' in batch.output:  # the sampler is responsible for constructing the correct shape
            # Sometimes, not every ray has the same number of samples
            sh = batch.output.z_vals.shape  # B, P, S
            rgb = rgb.view(sh + (-1,))  # B, P, S, 3
            occ = occ.view(sh + (-1,))  # B, P, S, 3

        # Use provided background color?
        bg_color = max(min(self.bg_brightness, 1.0), 0.0)  # TODO: Fill ground truth with this background color
        if self.training:
            if self.bg_brightness < 0:
                bg_color = torch.rand_like(rgb[..., 0, :])  # remove sample dim (in 0 -> 1 already)
            else:
                bg_color = torch.full_like(rgb[..., 0, :], self.bg_brightness)  # remove sample dim

        weights, rgb_map, acc_map = volume_rendering(rgb, occ, bg_color)  # B, P, S; B, P, 3; B, P
        output = dotdict()
        output.weights = weights  # for distortion loss computation
        output.rgb_map = rgb_map  # main output
        output.acc_map = acc_map  # with last dimension intact

        if isinstance(bg_color, torch.Tensor): output.bg_color = bg_color
        else: output.bg_color = torch.full_like(rgb_map, bg_color)  # B, P, 3

        # TODO: Differentiate between visualization and training
        # Lots of memory could be saved if only the needed parts are rendered out
        # For now, no depth or other fancy visualization to be rendered
        # Multiple colliding considerations:
        # 1. Extensible means we should not be configuring a lot of stuff
        # 2. Controllable means we'd like to omit unnecessary computation
        # 3. Type control vs GUI implementation switching?
        def integrate(value: torch.Tensor, weights: torch.Tensor = weights, bg_color: torch.Tensor = bg_color):
            value = (weights[..., None] * value.view(weights.shape + (-1,))).sum(dim=-2)
            value = value + (1. - acc_map) * bg_color  # B, P, 3
            return value

        # Normalize weight for further operation
        if self.normalize_weights:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)

        # Color output (normalized or no need for normalization)
        if 'norm' in batch.output:
            output.norm_map = integrate(batch.output.norm)  # B, P, 3

        # Position output
        if 'z_vals' in batch.output:
            # NOTE: Current guideline: if we need to concatenate, we retain last dim
            # output.dpt_map = integrate(batch.output.z_vals) / acc_map.clip(1e-6)  # without last dim
            if self.use_median_depth:
                output.dpt_map = weighted_percentile(torch.cat([batch.output.z_vals, batch.output.z_vals.max(dim=-1, keepdim=True)[0]], dim=-1),
                                                     torch.cat([weights, 1 - acc_map], dim=-1), [0.5])
            else:
                output.dpt_map = integrate(batch.output.z_vals, bg_color=batch.output.z_vals.max(dim=-1, keepdim=True)[0])  # without last dim

        if 'resd' in batch.output:
            output.resd_map = integrate(batch.output.resd)  # B, P, C

        if 'xyz' in batch.output:
            output.surf_map = integrate(batch.output.xyz)  # B, P, C

        # TODO: CHECK TYPES OF VISUALIZER
        # # Others
        # if 'xyzt_feat' in batch.output:
        #     output.feat_map = integrate(batch.output.xyzt_feat, bg_color=0.0)   # B, P, C

        # # TODO: Perform volume rendering for proposal networks
        # if 'rgb_prop' in batch.output:
        #     pass

        if self.network.training:
            batch.output.update(output)
        else:
            batch.output = output  # save some memory
        return rgb_map
