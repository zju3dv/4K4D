import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.loss_utils import eikonal
from easyvolcap.utils.net_utils import SphereSignedDistanceField, MLP, take_gradient
from lib.networks.embedder import get_embedder
# fmt: on

N_ITER = 2000
N = 4096
B = 4


class VLO_SDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.ebd, ebd_dim = get_embedder(6, 3)
        self.ebd_t, ebd_t_dim = get_embedder(12, 1)
        self.sdf = SphereSignedDistanceField(d_in=ebd_dim, d_out=1)
        self.vlo_mlp = MLP(input_ch=ebd_dim + ebd_t_dim, out_ch=3)

    def vlo(self, t, x) -> torch.Tensor:  # assuming already batched
        x = self.ebd(x)
        t = self.ebd_t(t)
        return self.vlo_mlp(torch.cat([x, t], dim=-1))

    def batch_vlo(self, t, x):
        t = t[None, None, None].expand(*x.shape[:2], -1)  # B, N, 1
        return self.vlo(t, x)

    def vlo_back(self, t, x, T=1):  # assuming already batched
        t = T - t
        return -self.vlo(t, x)

    def batch_vlo_back(self, t, xT):
        x, T = xT[:, :-1], xT[:, -1:, :1]  # extract the actual x value, B, N, 3; B, 1, 1
        t = t[None, None, None].expand(*x.shape[:2], -1)  # B, N, 1
        v = self.vlo_back(t, x, T)
        v = torch.cat([v, torch.zeros_like(T).expand(B, -1, 3)], dim=1)  # store T info as extra dim in batch
        return v

    forward = batch_vlo_back


vlo_sdf = VLO_SDF()
vlo_sdf.to('cuda')

o = Adam(vlo_sdf.parameters(), lr=5e-4)
p = tqdm(total=N_ITER)

for i in range(N_ITER):
    x = torch.rand(B, N, 3, device='cuda', requires_grad=True)
    x_observe = x

    T = torch.rand(B, device='cuda')  # every points has a different target timestep to evaluate to?
    t, t_idx = torch.unique(torch.cat([torch.zeros(1, device='cuda'), T]), sorted=True, return_inverse=True)
    t_idx = t_idx[1:]-1  # remove the first 0, and shift by 1
    xT = torch.cat([x, T[..., None, None].expand(B, -1, 3)], dim=1)

    # TODO: toooooooooooooo slow, 0.25 iter / s... whereas the other part of the model run in 0.25 s / iter
    # TODO: not double differentiable might be a huge problem to enforce eikonal term...
    # TODO: easily out of memory... unacceptably huge memory requirements for regular methods, while the adjoint method does not support double autodiff
    # TODO: let alone the fact that the adjoint method is terribly slow already on its own
    # TODO: even though we only apply eikonal term on near surface points, it's still a little bit slower
    # TODO: let alone the fact that double diff through this might not even produce correct results
    x = odeint(vlo_sdf, xT, t, rtol=1e-3, atol=1e-3)  # B, N, 3 # TODO: isn't this a bit wasteful? In terms of memory
    x = x[1:][torch.arange(len(t_idx)), t_idx]

    x = vlo_sdf.ebd(x)
    s = vlo_sdf.sdf(x)
    g = take_gradient(s, x_observe)
    l = eikonal(g)

    o.zero_grad(set_to_none=True)
    l.backward()
    o.step()

    p.update(1)
    p.desc = f'Loss: {l.item():.8f}'
