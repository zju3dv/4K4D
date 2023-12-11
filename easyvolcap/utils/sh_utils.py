# Modifications Copyright 2021 The PlenOctree Authors.
# Original Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sperical harmonics projection related functions

Some codes are borrowed from:
https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
"""
import math
import torch
from typing import Callable, Collection
from easyvolcap.utils.console_utils import *


def spher2cart(theta, phi):
    """Convert spherical coordinates into Cartesian coordinates (radius 1)."""
    r = torch.sin(theta)
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


# Get the total number of coefficients for a function represented by
# all spherical harmonic basis of degree <= @order (it is a point of
# confusion that the order of an SH refers to its degree and not the order).
def GetCoefficientCount(order: int):
    return int((order + 1) ** 2)


# Get the one dimensional index associated with a particular degree @l
# and order @m. This is the index that can be used to access the Coeffs
# returned by SHSolver.
def GetIndex(l: int, m: int):
    return l * (l + 1) + m


# Hardcoded spherical harmonic functions for low orders (l is first number
# and m is second number (sign encoded as preceeding 'p' or 'n')).
#
# As polynomials they are evaluated more efficiently in cartesian coordinates,
# assuming that @{dx, dy, dz} is unit. This is not verified for efficiency.

def HardcodedSH00(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.5 * sqrt(1/pi)
    return 0.28209479177387814 + (dx * 0)  # keep the shape


def HardcodedSH1n1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -sqrt(3/(4pi)) * y
    return -0.4886025119029199 * dy


def HardcodedSH10(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # sqrt(3/(4pi)) * z
    return 0.4886025119029199 * dz


def HardcodedSH1p1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -sqrt(3/(4pi)) * x
    return -0.4886025119029199 * dx


def HardcodedSH2n2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.5 * sqrt(15/pi) * x * y
    return 1.0925484305920792 * dx * dy


def HardcodedSH2n1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.5 * sqrt(15/pi) * y * z
    return -1.0925484305920792 * dy * dz


def HardcodedSH20(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
    return 0.31539156525252005 * (-dx * dx - dy * dy + 2.0 * dz * dz)


def HardcodedSH2p1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.5 * sqrt(15/pi) * x * z
    return -1.0925484305920792 * dx * dz


def HardcodedSH2p2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.25 * sqrt(15/pi) * (x^2 - y^2)
    return 0.5462742152960396 * (dx * dx - dy * dy)


def HardcodedSH3n3(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
    return -0.5900435899266435 * dy * (3.0 * dx * dx - dy * dy)


def HardcodedSH3n2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.5 * sqrt(105/pi) * x * y * z
    return 2.890611442640554 * dx * dy * dz


def HardcodedSH3n1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
    return -0.4570457994644658 * dy * (4.0 * dz * dz - dx * dx - dy * dy)


def HardcodedSH30(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
    return 0.3731763325901154 * dz * (2.0 * dz * dz - 3.0 * dx * dx - 3.0 * dy * dy)


def HardcodedSH3p1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
    return -0.4570457994644658 * dx * (4.0 * dz * dz - dx * dx - dy * dy)


def HardcodedSH3p2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
    return 1.445305721320277 * dz * (dx * dx - dy * dy)


def HardcodedSH3p3(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
    return -0.5900435899266435 * dx * (dx * dx - 3.0 * dy * dy)


def HardcodedSH4n4(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
    return 2.5033429417967046 * dx * dy * (dx * dx - dy * dy)


def HardcodedSH4n3(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
    return -1.7701307697799304 * dy * dz * (3.0 * dx * dx - dy * dy)


def HardcodedSH4n2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
    return 0.9461746957575601 * dx * dy * (7.0 * dz * dz - 1.0)


def HardcodedSH4n1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
    return -0.6690465435572892 * dy * dz * (7.0 * dz * dz - 3.0)


def HardcodedSH40(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
    z2 = dz * dz
    return 0.10578554691520431 * (35.0 * z2 * z2 - 30.0 * z2 + 3.0)


def HardcodedSH4p1(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
    return -0.6690465435572892 * dx * dz * (7.0 * dz * dz - 3.0)


def HardcodedSH4p2(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
    return 0.47308734787878004 * (dx * dx - dy * dy) * (7.0 * dz * dz - 1.0)


def HardcodedSH4p3(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
    return -1.7701307697799304 * dx * dz * (dx * dx - 3.0 * dy * dy)


def HardcodedSH4p4(dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor):
    # 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
    x2 = dx * dx
    y2 = dy * dy
    return 0.6258357354491761 * (x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2))


def EvalSH(l: int, m: int, dirs: torch.Tensor) -> torch.Tensor:
    """
    Args:
      dirs: array [..., 3]. works with torch/jnp/np
    Return:
      array [...]
    """
    # Validate l and m here (don't do it generally since EvalSHSlow also
    # checks it if we delegate to that function).
    assert l >= 0, "l must be at least 0."
    assert -l <= m and m <= l, "m must be between -l and l."
    dx = dirs[..., 0]
    dy = dirs[..., 1]
    dz = dirs[..., 2]

    if l == 0:
        return HardcodedSH00(dx, dy, dz)
    elif l == 1:
        if m == -1:
            return HardcodedSH1n1(dx, dy, dz)
        elif m == 0:
            return HardcodedSH10(dx, dy, dz)
        elif m == 1:
            return HardcodedSH1p1(dx, dy, dz)
        else:
            raise NotImplementedError
    elif l == 2:
        if m == -2:
            return HardcodedSH2n2(dx, dy, dz)
        elif m == -1:
            return HardcodedSH2n1(dx, dy, dz)
        elif m == 0:
            return HardcodedSH20(dx, dy, dz)
        elif m == 1:
            return HardcodedSH2p1(dx, dy, dz)
        elif m == 2:
            return HardcodedSH2p2(dx, dy, dz)
        else:
            raise NotImplementedError
    elif l == 3:
        if m == -3:
            return HardcodedSH3n3(dx, dy, dz)
        elif m == -2:
            return HardcodedSH3n2(dx, dy, dz)
        elif m == -1:
            return HardcodedSH3n1(dx, dy, dz)
        elif m == 0:
            return HardcodedSH30(dx, dy, dz)
        elif m == 1:
            return HardcodedSH3p1(dx, dy, dz)
        elif m == 2:
            return HardcodedSH3p2(dx, dy, dz)
        elif m == 3:
            return HardcodedSH3p3(dx, dy, dz)
        else:
            raise NotImplementedError
    elif l == 4:
        if m == -4:
            return HardcodedSH4n4(dx, dy, dz)
        elif m == -3:
            return HardcodedSH4n3(dx, dy, dz)
        elif m == -2:
            return HardcodedSH4n2(dx, dy, dz)
        elif m == -1:
            return HardcodedSH4n1(dx, dy, dz)
        elif m == 0:
            return HardcodedSH40(dx, dy, dz)
        elif m == 1:
            return HardcodedSH4p1(dx, dy, dz)
        elif m == 2:
            return HardcodedSH4p2(dx, dy, dz)
        elif m == 3:
            return HardcodedSH4p3(dx, dy, dz)
        elif m == 4:
            return HardcodedSH4p4(dx, dy, dz)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def spherical_uniform_sampling_upper(sample_count, device="cuda"):
    # See: https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta = torch.acos(1.0 * torch.rand([sample_count], device=device))
    phi = 2.0 * math.pi * torch.rand([sample_count], device=device)
    return theta, phi


def spherical_uniform_sampling(sample_count, device="cuda"):
    # See: https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta = torch.acos(2.0 * torch.rand([sample_count], device=device) - 1.0)
    phi = 2.0 * math.pi * torch.rand([sample_count], device=device)
    return theta, phi


@torch.jit.script
def project_values_least_squares(
    sh_deg: int,
    dirs: torch.Tensor,  # ..., 1, S, 3 -> ..., 1, S EvalSH
    vals: torch.Tensor   # ..., C, S
):
    C = 3  # rgb channels
    S = dirs.shape[-2]
    num_coeffs = GetCoefficientCount(sh_deg)

    # Calculate the spherical harmonic matrix for the given directions
    Y_shape = dirs.shape[:-3] + (S, num_coeffs,)  # ..., S, SH
    Y = dirs.new_empty(Y_shape)
    for l in range(sh_deg + 1):
        for m in range(-l, l + 1):
            Y[..., GetIndex(l, m)] = EvalSH(l, m, dirs)[..., 0, :]  # fill in ..., S

    # Now, we solve for the SH coefficients using least squares.
    # The least square's equation: vals.mT (S, C) = Y (S, SH) coeffs.mT (SH, C)
    # Solution for least squares is given by: coeffs.mT = (Y^T Y)^(-1) Y^T vals.mT (SH, S @ S, C)
    Yt = Y.mT  # ..., SH, S
    YtY = Yt @ Y  # ..., SH, SH
    YtY_inv = YtY.inverse()  # ..., SH, SH
    YtY_inv_Yt = YtY_inv @ Yt  # ..., SH, S
    coeffs = (YtY_inv_Yt @ vals.mT).mT  # ..., C, SH

    return coeffs  # coeffs: ..., C, SH


def project_values_least_squares_sparse(
    sh_deg: int,
    dirs: torch.Tensor,  # ..., 1, S, 3 -> ..., 1, S EvalSH
    vals: torch.Tensor,  # ..., C, S
    lr: float = 1.0,
    iter: int = 100,
    lambda_l1: float = 0.01,  # Regularization strength for L1 norm
    update_iter: int = 10,
    print_progress: bool = False,
):
    C = 3  # rgb channels
    S = dirs.shape[-2]
    num_coeffs = GetCoefficientCount(sh_deg)

    # Calculate the spherical harmonic matrix for the given directions
    Y_shape = dirs.shape[:-3] + (S, num_coeffs,)  # ..., S, SH
    Y = dirs.new_empty(Y_shape)
    for l in range(sh_deg + 1):
        for m in range(-l, l + 1):
            Y[..., GetIndex(l, m)] = EvalSH(l, m, dirs)[..., 0, :]  # fill in ..., S

    coeffs = torch.zeros(*Y.shape[:-2], C, num_coeffs, device=dirs.device)  # Initialize the coefficients as zeros for all channels
    coeffs.requires_grad = True  # We need to compute gradients with respect to coeffs

    optimizer = torch.optim.Adam([coeffs], lr=lr)  # Using Adam optimizer

    pbar = tqdm(range(iter), disable=not print_progress)
    for i in pbar:  # Number of optimization steps
        with torch.enable_grad():
            mse_loss = (Y @ coeffs.mT - vals.mT).pow(2).sum(dim=-1).mean()  # l2 loss
            l1_loss = lambda_l1 * coeffs.norm(p=1, dim=-1).mean()  # l1 loss
            loss = mse_loss + l1_loss
        if not (i + 1) % update_iter:
            pbar.desc = f'Loss: {loss.item():.6f}'  # MARK: SYNC
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return coeffs.detach()  # coeffs: ..., C, SH


@torch.jit.script
def project_values(
    sh_deg: int,
    dirs: torch.Tensor,  # N, 1, S, 3 -> N, 1, S EvalSH
    vals: torch.Tensor,  # N, 3, S
):
    """
    When we talk about the "energy" of a function, we're referring to the overall "strength" or "magnitude" of that function across its entire domain. Imagine the function's values squared and then summed up (or integrated) over its entire domain; that gives us the function's energy.

    For spherical harmonics, which are special functions defined on the surface of a sphere, there are two key properties:

    Orthogonality: This means that if you take any two different spherical harmonics and multiply their values point-by-point over the sphere, the sum of these products will be zero. In simpler terms, they don't overlap in a way that their combined strength adds up; instead, they cancel each other out.

    Normalization: This means that the energy of each individual spherical harmonic is 1. So, if you square its values and sum them up over the sphere, the total will be 1. It's like saying each spherical harmonic has a consistent strength or magnitude over the sphere.

    These properties make spherical harmonics very useful for representing complex functions on the sphere, as they provide a set of independent and consistent building blocks.
    """
    C = 3  # rgb channels

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.empty(vals.shape[:-2] + (C, GetCoefficientCount(sh_deg)), dtype=dirs.dtype, device=dirs.device)  # N, C, SH

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(sh_deg + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            coeffs[..., GetIndex(l, m)] = (vals * EvalSH(l, m, dirs)).sum(dim=-1)  # N, 1, S * N, C, S -> N, C, S -> N, C

    # scale by the probability of a particular sample, which is
    # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
    # 1/sample_count for the number of samples drawn uniformly.
    weight = 4.0 * math.pi / vals.shape[-1]
    coeffs *= weight
    return coeffs  # coeffs: N, C, SH


def project_function_least_squares(
    sh_deg: int,
    spherical_function: Callable,
    batch_size: int,
    sample_count: int,
    device="cuda"
):
    assert sh_deg >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.zeros([batch_size, C, GetCoefficientCount(sh_deg)], dtype=torch.float32, device=device)

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)  # SAM, 3

    # evaluate the analytic function for the current spherical coords
    vals = spherical_function(dirs).mT  # ..., SAM, C -> ..., C, SAM
    full = torch.zeros_like(vals)[..., None, :, :].mT  # ..., 1, SAM, C, C == 3
    full += dirs  # auto broadcast
    dirs = full

    C = 3  # rgb channels
    S = dirs.shape[-2]
    num_coeffs = GetCoefficientCount(sh_deg)

    # Calculate the spherical harmonic matrix for the given directions
    Y_shape = dirs.shape[:-3] + (S, num_coeffs,)  # ..., S, SH
    Y = dirs.new_empty(Y_shape)
    for l in range(sh_deg + 1):
        for m in range(-l, l + 1):
            Y[..., GetIndex(l, m)] = EvalSH(l, m, dirs)[..., 0, :]  # fill in ..., S

    # Now, we solve for the SH coefficients using least squares.
    # The least square's equation: vals.mT (S, C) = Y (S, SH) coeffs.mT (SH, C)
    # Solution for least squares is given by: coeffs.mT = (Y^T Y)^(-1) Y^T vals.mT (SH, S @ S, C)
    Yt = Y.mT  # ..., SH, S
    YtY = Yt @ Y  # ..., SH, SH
    YtY_inv = YtY.inverse()  # ..., SH, SH
    YtY_inv_Yt = YtY_inv @ Yt  # ..., SH, S
    coeffs = (YtY_inv_Yt @ vals.mT).mT  # ..., C, SH

    return coeffs  # coeffs: ..., C, SH


def project_function(
    sh_deg: int,
    spherical_function: Callable,
    batch_size: int,
    sample_count: int,
    device="cuda"
):
    assert sh_deg >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.zeros([batch_size, C, GetCoefficientCount(sh_deg)], dtype=torch.float32, device=device)

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)  # N, 3

    # evaluate the analytic function for the current spherical coords
    func_value = spherical_function(dirs)  # [batch_size, sample_count, C]

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(sh_deg + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            coeffs[:, :, GetIndex(l, m)] = torch.einsum("bsc,s->bc", func_value, EvalSH(l, m, dirs))

    # scale by the probability of a particular sample, which is
    # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
    # 1/sample_count for the number of samples drawn uniformly.
    weight = 4.0 * math.pi / sample_count
    coeffs *= weight
    return coeffs  # coeffs: n_samples, C, sh_dim


def ProjectFunction(
        order: int,
        sperical_func: Callable,
        sample_count: int,
        device="cuda"
):
    assert order >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.zeros([GetCoefficientCount(order)], dtype=torch.float32).to(device)

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)

    # evaluate the analytic function for the current spherical coords
    func_value = sperical_func(dirs)

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(order + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            coeffs[GetIndex(l, m)] = sum(func_value * EvalSH(l, m, dirs))

    # scale by the probability of a particular sample, which is
    # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
    # 1/sample_count for the number of samples drawn uniformly.
    weight = 4.0 * math.pi / sample_count
    coeffs *= weight
    return coeffs


def ProjectFunctionNeRFReuse(
    order: int,
    spherical_function: Callable[[torch.Tensor], Collection[torch.Tensor]],
    n_points: int,
    n_samples: int,
    n_rays: int,
    device="cpu"
):
    assert order >= 0, "Order must be at least zero."
    assert n_samples * n_rays > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.zeros([n_points, C, GetCoefficientCount(order)], dtype=torch.float32, device=device)

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(n_samples * n_rays, device=device)
    dirs = spher2cart(theta, phi)

    # evaluate the analytic function for the current spherical coords
    func_value, others = spherical_function(dirs.view(n_samples, n_rays, -1))  # [n_points, n_samples, n_rays, C], [n_points, n_samples, 1]
    func_value = func_value.view(n_points, -1, C)  # n_points, n_samples * n_rays, 3
    others = others.mean(dim=1)  # take mean of other samples

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(order + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            coeffs[:, :, GetIndex(l, m)] = torch.einsum("bsc,s->bc", func_value, EvalSH(l, m, dirs))

    # scale by the probability of a particular sample, which is
    # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
    # 1/sample_count for the number of samples drawn uniformly.
    weight = 4.0 * math.pi / (n_rays * n_samples)
    coeffs *= weight
    return coeffs, others  # coeffs: n_samples, C, sh_dim; others


def ProjectFunctionNeRF(
    order: int,
    spherical_function: Callable,
    batch_size: int,
    sample_count: int,
    device="cpu"
):
    assert order >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # This is the approach demonstrated in [1] and is useful for arbitrary
    # functions on the sphere that are represented analytically.
    coeffs = torch.zeros([batch_size, C, GetCoefficientCount(order)], dtype=torch.float32, device=device)

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)

    # evaluate the analytic function for the current spherical coords
    func_value, others = spherical_function(dirs)  # [batch_size, sample_count, C]

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(order + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            coeffs[:, :, GetIndex(l, m)] = torch.einsum("bsc,s->bc", func_value, EvalSH(l, m, dirs))

    # scale by the probability of a particular sample, which is
    # 4pi/sample_count. 4pi for the surface area of a unit sphere, and
    # 1/sample_count for the number of samples drawn uniformly.
    weight = 4.0 * math.pi / sample_count
    coeffs *= weight
    return coeffs, others  # coeffs: n_samples, C, sh_dim; others


def ProjectFunctionNeRFSparse(
    order: int,
    spherical_function: Callable,
    sample_count: int,
    device="cpu",
):
    assert order >= 0, "Order must be at least zero."
    assert sample_count > 0, "Sample count must be at least one."
    C = 3  # rgb channels

    # generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling(sample_count, device=device)
    dirs = spher2cart(theta, phi)  # [sample_count, 3]

    # evaluate the analytic function for the current spherical coords
    func_value, others = spherical_function(dirs)  # func_value [batch_size, sample_count, C]

    batch_size = func_value.shape[0]

    coeff_count = GetCoefficientCount(order)
    basis_vals = torch.empty([sample_count, coeff_count], dtype=torch.float32, device=device)

    # evaluate the SH basis functions up to band O, scale them by the
    # function's value and accumulate them over all generated samples
    for l in range(order + 1):  # end inclusive
        for m in range(-l, l + 1):  # end inclusive
            basis_vals[:, GetIndex(l, m)] = EvalSH(l, m, dirs)

    basis_vals = basis_vals.view(sample_count, coeff_count)  # [sample_count, coeff_count]
    func_value = func_value.transpose(0, 1).reshape(sample_count, batch_size * C)  # [sample_count, batch_size * C]
    soln = torch.lstsq(func_value, basis_vals).solution[:basis_vals.size(1)]
    soln = soln.T.reshape(batch_size, C, -1)
    return soln, others

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.


@torch.jit.script
def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]

    Returns:
        [..., C]
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])
            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy) * sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])
                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                              C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                              C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                              C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                              C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                              C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                              C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                              C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                              C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


if __name__ == '__main__':
    import numpy as np
    deg = 3
    sh = np.random.random((400, 400, 3, 16))
    dirs = np.random.random((400, 400, 3))
    result = eval_sh(deg, sh, dirs)
    print(result.shape)
