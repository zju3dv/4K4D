
import torch
from torch import nn

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.net_utils import MLP, Modulized, get_function
from easyvolcap.utils.base_utils import dotdict


@REGRESSORS.register_module()
class SplitRegressor(nn.Module):
    # Outputs occupancy (used as alpha in volume rendering)
    def __init__(self,
                 in_dim: int,
                 width: int = 256,
                 depth: int = 8,

                 splits=[1, 256],
                 #  activs=nn.ModuleList([Modulized(raw2alpha), nn.Identity()])
                 #  activs=nn.ModuleList([nn.Sigmoid(), nn.Identity()])
                 #  activs=nn.ModuleList([nn.ReLU(), nn.Identity()])
                 activs=nn.ModuleList([nn.Softplus(), nn.Identity()]),
                 sequential_split=False,
                 dtype: str = 'float',
                 **kwargs,
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = sum(splits)  
        self.width = width
        self.depth = depth
        self.sequential_split = sequential_split

        if sequential_split:
            assert len(splits) == 2  # only support two levels of split for now, note that the previous layer is ReLUed
            assert splits[-1] == width  # NOTE: not considering skip connection
            mlp_out_dim = splits[0]
        else:
            mlp_out_dim = self.out_dim

        self.mlp = MLP(in_dim, width, depth, mlp_out_dim, dtype=dtype)
        self.splits = splits
        self.activs = nn.ModuleList([get_function(activ) for activ in activs])

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        if not self.sequential_split:
            feat: torch.Tensor = self.mlp(feat)
            feat = feat.split(self.splits, dim=-1)
        else:
            feat: torch.Tensor = self.mlp.forward_with_previous(feat)
        feat = [self.activs[i](f) for i, f in enumerate(feat)]  # occ, feature
        feat = torch.cat(feat, dim=-1)
        return feat
