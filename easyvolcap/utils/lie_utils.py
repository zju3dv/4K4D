# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper for Lie group operations. Currently only used for pose optimization.
"""
import torch


# We make an exception on snake case conventions because SO3 != so3.
@torch.jit.script
def exp_map_SO3xR3(tangent_vector: torch.Tensor) -> torch.Tensor:
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[..., 3:]  # B, N, 3
    norms = (log_rot * log_rot).sum(-1)  # B, N
    rot_angles = torch.clamp(norms, 1e-4).sqrt()  # B, N
    rot_angles_inv = 1.0 / rot_angles  # B, N
    fac1 = rot_angles_inv * rot_angles.sin()  # B, N
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())  # B, N
    skews = torch.zeros(tangent_vector.shape[:-1] + (3, 3), dtype=log_rot.dtype, device=log_rot.device)  # B, N, 3, 3
    # fmt: off
    skews[..., 0, 1] = -log_rot[..., 2]
    skews[..., 0, 2] =  log_rot[..., 1]
    skews[..., 1, 0] =  log_rot[..., 2]
    skews[..., 1, 2] = -log_rot[..., 0]
    skews[..., 2, 0] = -log_rot[..., 1]
    skews[..., 2, 1] =  log_rot[..., 0]
    # fmt: on
    skews_square = skews @ skews  # B, N, 3, 3

    ret = torch.zeros(tangent_vector.shape[:-1] + (3, 4), dtype=tangent_vector.dtype, device=tangent_vector.device)  # B, N, 3, 4
    ide = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device) # 3, 3
    for s in tangent_vector.shape[:-1][::-1]:
        ide = ide[None].expand((s,) + ide.shape)
    ret[..., :3, :3] = (
        fac1[..., None, None] * skews
        + fac2[..., None, None] * skews_square
        + ide
    )

    # Compute the translation
    ret[..., :3, 3] = tangent_vector[..., :3]  # B, N, 3, 4
    return ret


@torch.jit.script
def exp_map_SE3(tangent_vector: torch.Tensor) -> torch.Tensor:
    # TODO: Modify this to support multiple batch dimensions
    """Compute the exponential map `se(3) -> SE(3)`.

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`.

    Returns:
        [R|t] transformation matrices.
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret
