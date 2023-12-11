import torch
from torch import nn

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.net_utils import MLP
from easyvolcap.utils.base_utils import dotdict


@REGRESSORS.register_module()
class TcnnMlpRegressor(nn.Module):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 3,
                 width: int = 256,
                 depth: int = 8,
                 otype: str = 'FullyFusedMLP',
                 actvn: str = 'ReLU',
                 out_actvn: str = 'Sigmoid',
                 dtype: str = torch.float,  # FIXME: tcnn mlp doesn't support configuring data types
                 ):
        # Simply an MLP wrapper
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth

        network_config = dotdict()
        network_config.otype = otype
        network_config.activation = actvn
        network_config.output_activation = out_actvn
        network_config.n_neurons = width
        network_config.n_hidden_layers = depth - 1  # throw std::runtime_error("FullyFusedMLP requires at least 1 hidden layer (3 layers in total).");

        import tinycudann as tcnn  # lasy imports
        self.mlp = tcnn.Network(n_input_dims=in_dim,
                                n_output_dims=out_dim,
                                network_config=network_config)

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        return self.mlp(feat.view(-1, feat.shape[-1])).view(*feat.shape[:-1], -1)  # no sequential and forward_with_previous split
