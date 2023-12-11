import torch
from torch import nn
from easyvolcap.utils.net_utils import MLP, GradientModule
from lib.networks.embedder import get_embedder

from easyvolcap.engine import cfg


def parameterize(x: torch.Tensor, v: torch.Tensor):
    s = (-x * v).sum(dim=-1, keepdim=True)  # closest distance along ray to origin
    o = s * v + x  # closest point along ray to origin
    return o, s


class DirectionalDistance(GradientModule):
    def __init__(self,
                 xyz_res=8,  # use middle resolution?
                 view_res=8,  # are there still view encoding needs?
                 cond_dim=cfg.cond_dim,
                 ):
        super(DirectionalDistance, self).__init__()
        # our directional distance module for fast sphere tracing (pre-computation)
        # before learning them in a brute-force way, we need to analyze the properties of these modules
        # for directional distance, occ along the same ray should stay the same, we might need a specific network structure to ensure that
        # for directional intersection, distance to intersection point along the same ray should be constrained by a ray equation, too needs special structure
        # is it possible to define a neural field parameterized by ray instead of points and directions?
        # * a good enough parameterization should be: find closest points along ray, compute through network and then convert

        # this module stores the closest distance along a ray direction from a point in space along a ray direction
        self.xyz_embedder, xyz_dim = get_embedder(xyz_res, 3)  # no parameters
        self.view_embedder, view_dim = get_embedder(view_res, 3)  # no parameters
        # self.mlp = MLP(input_ch=xyz_dim + view_dim + cond_dim, W=256, D=8, out_ch=1 + 1, actvn=nn.Softplus(), out_actvn=nn.Identity())
        self.directional_distance = MLP(input_ch=xyz_dim + view_dim + cond_dim, W=256, D=8, out_ch=1, actvn=nn.Softplus(), out_actvn=nn.Identity())
        # this module stores the surface intersection point distance along a ray direction from a point in space along a ray direction
        self.directional_intersection = MLP(input_ch=xyz_dim + view_dim + cond_dim, W=256, D=8, out_ch=1, actvn=nn.Softplus(), out_actvn=nn.Identity())
        # self.intersection_probability = MLP(input_ch=xyz_dim + view_dim + cond_dim, W=256, D=8, out_ch=1, actvn=nn.Softplus(), out_actvn=nn.Sigmoid())

    def forward(self, x: torch.Tensor, v: torch.Tensor, c: torch.Tensor):

        # maybe expand condition vector
        if c.ndim == 2:
            c = c[: None].expand(*v.shape[:2], -1)

        # find parameterization for a particular ray
        o, s = parameterize(x, v)  # origin distance and origin intersection

        # forward through the network
        ebd_o = self.xyz_embedder(o)
        ebd_v = self.view_embedder(v)
        input = torch.cat([ebd_o, ebd_v, c], dim=-1)
        # out = self.mlp(input)
        # dd, di = out.split([1, 1], dim=-1)
        dd = self.directional_distance(input)
        di = self.directional_intersection(input)
        # dd, di, pi = out.split([1, 1, 1], dim=-1)
        dd = dd.tanh()  # one meter for closest distance should be enough?
        di = di.tanh() * cfg.clip_far * 2  # larger range for intersection distance (to cover far plane)
        # pi = pi.sigmoid()

        # intersection_mask = pi > 0.5  # MARK: GRAD
        # dd = ~intersection_mask * dd  # not intersection -> use original value, intersection -> zero
        # di = pi * di + ~intersection_mask * cfg.clip_far  # intersection -> use original value, not intersection -> use far
        di = di + s  # plus the distance along ray to origin

        return dd, di
