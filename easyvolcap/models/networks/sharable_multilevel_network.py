import copy
import torch
from torch import nn
from typing import List, Tuple
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.engine import NETWORKS, EMBEDDERS, REGRESSORS

from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.networks.multilevel_network import MultilevelNetwork
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork
from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor
from easyvolcap.models.networks.regressors.empty_regressor import EmptyRegressor
from easyvolcap.models.networks.regressors.split_regressor import SplitRegressor
from easyvolcap.models.networks.regressors.se3_regressor import SE3Regressor
from easyvolcap.models.networks.regressors.noop_regressor import NoopRegressor

from easyvolcap.models.networks.embedders.empty_embedder import EmptyEmbedder
from easyvolcap.models.networks.embedders.positional_encoding_embedder import PositionalEncodingEmbedder
from easyvolcap.models.networks.embedders.composed_xyzt_embedder import ComposedXyztEmbedder


@NETWORKS.register_module()
class SharableMultilevelNetwork(MultilevelNetwork):
    def __init__(self,
                 network_cfgs: List[dotdict] = [
                     dotdict(
                         type=VolumetricVideoNetwork.__name__,
                         deformer_cfg=dotdict(
                             type=EmptyRegressor.__name__,
                         ),
                         geometry_cfg=dotdict(
                             type=SplitRegressor.__name__,
                             width=128, depth=4
                         ),
                         appearance_cfg=dotdict(
                             type=EmptyRegressor.__name__,
                         ),
                     ),  # coarse
                     dotdict(
                         type=VolumetricVideoNetwork.__name__,
                         deformer_cfg=dotdict(
                             type=EmptyRegressor.__name__,
                         ),
                         geometry_cfg=dotdict(
                             type=SplitRegressor.__name__,
                             width=512, depth=8
                         ),
                         appearance_cfg=dotdict(
                             type=MlpRegressor.__name__,
                             width=256, depth=2, out_dim=3),
                     ),  # fine
                 ],
                 shared_embedder_cfgs: dotdict = dotdict(
                     xyzt_embedder_cfg=dotdict(
                         type=ComposedXyztEmbedder.__name__,
                     ),
                     xyz_embedder_cfg=dotdict(
                         type=PositionalEncodingEmbedder.__name__,
                         multires=8,
                     ),
                     rgb_embedder_cfg=dotdict(
                         type=EmptyEmbedder.__name__,
                     ),
                     dir_embedder_cfg=dotdict(
                         type=PositionalEncodingEmbedder.__name__,
                         multires=4,
                     ),
                 ),
                 shared_regressor_cfgs: dotdict = dotdict(
                     parameterizer_cfg=dotdict(
                         type=NoopRegressor.__name__,
                     ),
                     deformer_cfg=dotdict(
                         type=SE3Regressor.__name__,
                         width=128, depth=6,
                     ),
                 ),
                 **kwargs,  # will feed these into the lower level networks
                 ):
        # What we called modularization
        call_from_cfg(super().__init__, kwargs, network_cfgs=network_cfgs)

        # build shared embedders
        self.shared_embedders: dotdict = dotdict({
            key.replace('_cfg', ''): EMBEDDERS.build(copy.deepcopy(kwargs[key]).update(value))
            for key, value in shared_embedder_cfgs.items()
        })

        # build shared regressors
        self.shared_regressors: dotdict = dotdict({
            key.replace('_cfg', ''): REGRESSORS.build(copy.deepcopy(kwargs[key]).update(value), in_dim=getattr(self.networks[0], key.replace('_cfg', '')).in_dim)
            for key, value in shared_regressor_cfgs.items()
        })

        # update the corresponding embedders and regressors in the networks
        for network in self.networks:
            for key, value in self.shared_embedders.items(): setattr(network, key, value)
            for key, value in self.shared_regressors.items(): setattr(network, key, value)
