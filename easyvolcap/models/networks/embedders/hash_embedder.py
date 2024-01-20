import torch
import numpy as np

from torch import nn
from sympy import nextprime
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import make_buffer, make_params
from easyvolcap.engine import EMBEDDERS
from easyvolcap.engine import cfg
from typing import List


@EMBEDDERS.register_module()
class HashEmbedder(nn.Module):
    from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera

    def __init__(self,
                 n_levels=16,
                 n_features_per_level=2,
                 b=1.38,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 sum=False,
                 sum_over_features=False,
                 separate_dense=True,
                 include_input=True,  # this will pass gradient better to input, but if you're using uvt, no need
                 ps=[1, 19349663, 83492791, 166985587],

                 bounds: List[List[int]] = OptimizableCamera.square_bounds,
                 in_dim: int = 3,
                 predefined_sizes: List[int] = [-1, -1, -1],  # this overwrites the computed size from b, level and base
                 ):
        """
        best iter speed: separate_dense = True
        best performace: separate_dense = False, sum_over_features = True
        """
        super().__init__()
        self.t = log2_hashmap_size
        self.n_levels = n_levels
        self.include_input = include_input
        self.n_entries_per_level = nextprime(2**log2_hashmap_size)
        self.predefined_sizes = predefined_sizes

        self.ps = [1]  # enough primes
        for i in range(1, in_dim):
            if i < len(ps): self.ps.append(ps[i])
            else: self.ps.append(nextprime(self.ps[-1] * 2))

        self.b = b
        self.f = n_features_per_level
        self.base_resolution = base_resolution

        self.bounds = make_buffer(torch.as_tensor(bounds, dtype=torch.float32))

        # every level should have this number of entries per side
        # we'd like the border to be mapped inside 0, 1
        self.entries_num = [[int((self.base_resolution * self.b**i)) if predefined_sizes[j] == -1 else predefined_sizes[j] for j in range(in_dim)] for i in range(self.n_levels)]  # L, 3
        self.entries_cnt = [np.prod(self.entries_num[i]) for i in range(self.n_levels)]  # L,
        self.entries_size = [1 / (np.asarray(self.entries_num[i]) - 1) for i in range(self.n_levels)]  # L, 3
        self.entries_min = [[0 for i in range(in_dim)] for i in range(self.n_levels)]  # L, 3

        self.entries_size = make_buffer(torch.as_tensor(self.entries_size, dtype=torch.float))  # L, 3
        self.entries_num = make_buffer(torch.as_tensor(self.entries_num, dtype=torch.long))  # L, 3
        self.entries_min = make_buffer(torch.as_tensor(self.entries_min, dtype=torch.long))  # L, 3
        self.entries_cnt = make_buffer(torch.as_tensor(self.entries_cnt, dtype=torch.long))  # L,
        self.entries_sum = make_buffer(self.entries_cnt.cumsum(dim=-1))  # L,

        self.start_hash = self.n_levels
        for i in range(n_levels):
            if self.entries_cnt[i] > self.n_entries_per_level:
                self.start_hash = i
                break
        self.len_hash = self.n_levels - self.start_hash
        self.separate_dense = separate_dense and self.start_hash  # when everything needs to be hashed for example when body using using small table
        if self.separate_dense:
            data = torch.zeros((self.n_levels, self.n_entries_per_level, self.f))
            # nn.init.kaiming_normal_(data)  # NOTE: initialization matters! separate_dense doesn't work well if we initialize the self.dense and self.hash data separately
            nn.init.uniform_(data, -1e-4, 1e-4)
            dense = torch.cat([data[i, :self.entries_cnt[i], :] for i in range(self.start_hash)], dim=0)
            hash = data[self.start_hash:, :, :]
            self.dense = make_params(dense)  # sum(non-hash), F
            self.hash = make_params(hash)  # H, T, F
        else:
            self.hash = make_params(torch.zeros((self.n_levels, self.n_entries_per_level, self.f)))  # H, T, F
            nn.init.uniform_(self.hash, -1e-4, 1e-4)

        # Input dim aware offset preparation
        offsets = []
        for i in range(2**in_dim):
            number = [0 for j in range(in_dim)]  # number of digits
            for j in range(in_dim - 1, -1, -1):  # in_dim-1, ..., 0
                if i >= 2**j:
                    i = i - 2**j  # remove this digit
                    number[in_dim - 1 - j] = 1  # fill this digit
                    if i == 2**j:
                        break
            offsets.append(number)
        self.offsets = make_buffer(torch.as_tensor(offsets, dtype=torch.float))

        self.sum = sum
        self.sum_over_features = sum_over_features

        self.in_dim = in_dim
        self.out_dim = 0
        if self.sum:
            if self.sum_over_features:
                self.out_dim += self.n_levels
            else:
                self.out_dim += self.f
        else:
            self.out_dim += self.f * self.n_levels

        if include_input:
            self.out_dim += in_dim

    def forward(self, xyz: torch.Tensor, batch: dotdict = None):
        bash = xyz.shape  # batch shape
        xyz = xyz.view(np.prod(bash[:-1]), xyz.shape[-1])

        N, _ = xyz.shape  # N, 3
        xyz = (xyz - self.bounds[0]) / (self.bounds[1] - self.bounds[0])  # normalized, N, 3

        ind_xyz = xyz[None].expand(self.n_levels, -1, -1)  # L, N, 3
        flt_xyz = ind_xyz / self.entries_size[:, None]  # L, N, 3
        int_xyz = (flt_xyz[:, :, None] + self.offsets[None, None]).long()  # will round to zero, L, N, 8, 3
        int_xyz = int_xyz.clip(self.entries_min[:, None, None], self.entries_num[:, None, None] - 1)
        off_xyz = flt_xyz - int_xyz[:, :, 0]  # L, N, 3

        sh = self.start_hash
        nl = self.n_levels

        # x as first digit, y as second digit, z as last digit -> S, N, 8
        ind_dense = torch.zeros_like(int_xyz[:sh, ..., 0])
        for i in range(self.in_dim):
            # All indices are treated as numbers
            # The first digit is the largest, the last digit is the smallest
            # Should also respect the hand-crafted size
            # L, 3 -> S, dim -> S, 1, 1 -> S, N, 8
            ind_dense = ind_dense + int_xyz[:sh, ..., i] * torch.prod(self.entries_num[:sh, i + 1:][:, None], dim=-1, keepdim=True)
        if self.separate_dense:
            ind_dense[1:] = ind_dense[1:] + self.entries_sum[:self.start_hash - 1][:, None, None]  # S, N, 8

        # hashing -> H, N, 8
        ind_hash = torch.ones_like(int_xyz[sh:, ..., 0])
        for i in range(self.in_dim):
            ind_hash = ind_hash ^ int_xyz[sh:, ..., i] * self.ps[i]  # doesn't matter
        ind_hash = ind_hash % self.n_entries_per_level
        if not self.separate_dense:
            ind = torch.cat([ind_dense, ind_hash], dim=0)

        # data: L, T, F, ind: L, N, 8 -> L, N, 8, F feature
        # NOTE: gather backward is much faster than index_select
        # val = self.data[torch.arange(nl, dtype=torch.long, device=ind.device)[..., None, None], ind, :]  # -> L, N, 8, F
        L, T, F = self.n_levels, self.n_entries_per_level, self.f
        S, H = self.start_hash, self.n_levels - self.start_hash
        K = 2 ** self.in_dim

        if self.separate_dense:
            val_dense = self.dense.gather(dim=0, index=ind_dense.view(S * N * K)[..., None].expand(-1, F)).view(S, N, K, F)
            val_hash = self.hash.gather(dim=1, index=ind_hash.view(H, N * K)[..., None].expand(-1, -1, F)).view(H, N, K, F)
            val = torch.cat([val_dense, val_hash], dim=0)
        else:
            val = self.hash.gather(dim=1, index=ind.view(L, N * K)[..., None].expand(-1, -1, F)).view(L, N, K, F)

        # off: L, N, 3, sets: 8, 3 -> L, N, :, 3 and :, :, 8, 3, compute xyz distance to the other corner, mul: multiplier
        mul_xyz = (1 - self.offsets[None, None]) + (2 * self.offsets[None, None] - 1.) * off_xyz[:, :, None]
        mul_xyz = mul_xyz[..., 0] * mul_xyz[..., 1]  # L, N, 8
        val = (mul_xyz[..., None] * val).sum(dim=-2)  # trilinear interpolated feature, L, N, F

        # feature aggregation
        val = val.permute(1, 0, 2)  # N, L, F
        if self.sum:
            if self.sum_over_features:
                val = val.sum(dim=-1)  # N, F, NOTE: sum over features seems to be producing better results...
            else:
                val = val.sum(dim=-2)  # N, L, NOTE: sum over features seems to be producing better results...
        else:
            val = val.reshape(-1, L * F)  # N, L*F

        # feature boosting
        if self.include_input:
            val = torch.cat([xyz, val], dim=-1)

        val = val.view(*bash[:-1], val.shape[-1])
        return val

    def extra_repr(self) -> str:
        # will be visible in print
        self.extra_dict = dotdict()
        self.extra_dict.base_resolution = self.base_resolution
        self.extra_dict.n_levels = self.n_levels
        # self.extra_dict.include_input = self.include_input
        self.extra_dict.t = self.t
        # self.extra_dict.ps = self.ps
        self.extra_dict.b = self.b
        self.extra_dict.f = self.f

        self.extra_dict.in_dim = self.in_dim
        self.extra_dict.predefined_sizes = self.predefined_sizes

        return ', '.join([k + '=' + str(v) for k, v in self.extra_dict.items()])
