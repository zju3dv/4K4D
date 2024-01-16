# Default optimizable parameter pipeline for volumetric videos
# Since it's volumetric, we define the 6d plenoptic function
# So no funny ray parameterization here (plucker, etc)
# This should be the most general purpose implementation
# parameterization (xyztθφ -> xyztθφ) ->
# xyzt embedder (xyzt -> feat) ->
# deformer (xyztθφ, feat -> xyztθφ) ->
# xyz embedder (xyzt -> feat) ->
# geometry (feat -> occ, feat) ->
# tθφ embedder (tθφ -> feat) ->
# appearance (feat, feat -> rgb)

import torch
from easyvolcap.engine import NETWORKS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import NoopModule


@NETWORKS.register_module()
class NoopNetwork(NoopModule):
    # fmt: off
    def __init__(self,
                 **kwargs, # suppress warnings
                 ):
        super().__init__()
        self.forward = self.compute

    def compute(self,
                xyz: torch.Tensor, dir: torch.Tensor, t: torch.Tensor, dist: torch.Tensor,
                batch: dotdict):
        # xyz: B, P, 3
        # dir: B, P, 3
        # t: B, P, 1
        # batch: dotdict
        # output: dotdict, output from sampler, should integrate on this

        # This pipeline should cover most of the cases
        # So try to only change its inner part instead of the whole pipeline
        # Unless you're looking for doing funny things like stream training or meta-learning

        return None, None
