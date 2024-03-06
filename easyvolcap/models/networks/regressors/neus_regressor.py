import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import get_function


@REGRESSORS.register_module()
class SDFRegressor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 width: int = 256,
                 depth: int = 8,
                 splits=[1, 256],
                 bias: float = 0.5,
                 skip_in: int = 4,
                 geometric_init: bool = True,
                 inside_outside: bool = False,
                 weight_norm: bool = True,
                 activs=nn.ModuleList([nn.Identity(), nn.Identity()]),
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = sum(splits)
        self.splits = splits
        self.bias = bias
        self.geometric_init = geometric_init
        self.inside_outside = inside_outside
        self.weight_norm = weight_norm
        self.activs = activs

        dims = [width for _ in range(depth)]
        dims = [self.in_dim] + dims + [self.out_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [skip_in] if isinstance(skip_in, int) else skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.geometric_init:
                if l == self.num_layers - 2:
                    if not self.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        x = feat

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, feat], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        feat = x.split(self.splits, dim=-1)
        feat = [self.activs[i](f) for i, f in enumerate(feat)]
        feat = torch.cat(feat, dim=-1)
        return feat


@REGRESSORS.register_module()
class ColorRegressor(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim: int = 3,
                 width: int = 256,
                 depth: int = 4,
                 weight_norm: bool = True,
                 out_actvn='sigmoid',
                 rgb_padding: float = 0.001
                 ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_norm = weight_norm
        self.rgb_padding = rgb_padding

        dims = [width for _ in range(depth)]
        dims = [in_dim] + dims + [3]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            nn.init.kaiming_uniform_(lin.weight)
            nn.init.zeros_(lin.bias.data)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "lin" + str(l), lin)

        self.actvn = nn.ReLU()
        self.out_actvn = get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn

    def forward(self, feat: torch.Tensor, batch: dotdict = None):
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, 'lin' + str(l))

            feat = lin(feat)

            if l < self.num_layers - 2:
                feat = self.actvn(feat)

        feat = self.out_actvn(feat)
        feat = feat * (1 + 2 * self.rgb_padding) - self.rgb_padding
        return feat


@REGRESSORS.register_module()
class SingleVarianceRegressor(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceRegressor, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@REGRESSORS.register_module()
class LaplaceDensityRegressor(nn.Module):
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""
        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


@REGRESSORS.register_module()
class SigmoidDensityRegressor(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, sdf, beta=None):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta
