import torch
from torch import nn

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.net_utils import MLP
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.blend_utils import screw2rt


@REGRESSORS.register_module()
class SE3Regressor(nn.Module):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 in_dim: int,
                 out_dim: int = 6,
                 width: int = 256,
                 depth: int = 8,
                 actvn: int = nn.ReLU(),
                 out_actvn: int = nn.Identity(),
                 ):
        # Simply an MLP wrapper
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth

        self.mlp = MLP(in_dim, width, depth, out_dim, actvn=actvn, out_actvn=out_actvn,
                       init_weight=lambda x: nn.init.xavier_uniform_(x), init_out_weight=lambda x: nn.init.uniform_(x, a=0.0, b=1e-4),
                       init_bias=lambda x: nn.init.zeros_(x), init_out_bias=lambda x: nn.init.zeros_(x))

    def forward(self, feature: torch.Tensor, batch: dotdict = None):
        """ Pass the embedded xyzt feature to the actual deform field and get SE3 screw axis output S=(r; v) ∈ R^6.
        Args:
            feature (torch.Tensor): (B, N, embed_dim) embeded xyzt features, xyz embedding + latent code embedding.
            batch (dict): stores the metadata and previous outputs.
        Returns:
            resd (torch.Tensor): (B, N, 3), residual displacements for each sample point.
        """
        screw = self.mlp(feature)   # (B, N, 6)
        # convert a batch of logarithmic representations of SE(3) matrices `log_transform`
        # to a batch of 4x4 SE(3) matrices using the exponential map
        rt = screw2rt(screw)        # (B, N, 6)
        return rt
