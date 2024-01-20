import torch


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script
def normalize_sum(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.sum(dim=-1, keepdim=True) + eps)

# Strange synchronization here if using torch.jit.script


@torch.jit.script
def point_padding(v: torch.Tensor):
    pad = torch.zeros_like(v[..., -1:])
    pad[..., -1] = 1.0
    ext = torch.cat([v, pad], dim=-1)
    return ext


@torch.jit.script
def vector_padding(v: torch.Tensor):
    pad = torch.zeros_like(v[..., -1:])
    ext = torch.cat([v, pad], dim=-1)
    return ext


@torch.jit.script
def affine_padding(c2w: torch.Tensor):
    # Already padded
    if c2w.shape[-2] == 4:
        return c2w
    # Batch agnostic padding
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))  # B, 1, 3
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))  # B, 1, 1
    pad = torch.cat([pad0, pad1], dim=-1)  # B, 1, 4
    c2w = torch.cat([c2w, pad], dim=-2)  # B, 4, 4
    return c2w


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)

# these works with an extra batch dimension
# Batched inverse of lower triangular matrices

# @torch.jit.script


def torch_trace(x: torch.Tensor):
    return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


# @torch.jit.script
def torch_inverse_decomp(L: torch.Tensor, eps=1e-10):
    n = L.shape[-1]
    invL = torch.zeros_like(L)
    for j in range(0, n):
        invL[..., j, j] = 1.0 / (L[..., j, j] + eps)
        for i in range(j + 1, n):
            S = 0.0
            for k in range(i + 1):
                S = S - L[..., i, k] * invL[..., k, j].clone()
            invL[..., i, j] = S / (L[..., i, i] + eps)

    return invL


def torch_inverse_3x3_precompute(R: torch.Tensor, eps=torch.finfo(torch.float).eps):
    # B, N, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """

    if not hasattr(torch_inverse_3x3_precompute, 'g_idx_i'):
        g_idx_i = torch.tensor(
            [
                [
                    [[1, 1], [2, 2]],
                    [[1, 1], [2, 2]],
                    [[1, 1], [2, 2]],
                ],
                [
                    [[0, 0], [2, 2]],
                    [[0, 0], [2, 2]],
                    [[0, 0], [2, 2]],
                ],
                [
                    [[0, 0], [1, 1]],
                    [[0, 0], [1, 1]],
                    [[0, 0], [1, 1]],
                ],
            ], device='cuda', dtype=torch.long)

        g_idx_j = torch.tensor(
            [
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
                [
                    [[1, 2], [1, 2]],
                    [[0, 2], [0, 2]],
                    [[0, 1], [0, 1]],
                ],
            ], device='cuda', dtype=torch.long)

        g_signs = torch.tensor([
            [+1, -1, +1],
            [-1, +1, -1],
            [+1, -1, +1],
        ], device='cuda', dtype=torch.long)

        torch_inverse_3x3_precompute.g_idx_i = g_idx_i
        torch_inverse_3x3_precompute.g_idx_j = g_idx_j
        torch_inverse_3x3_precompute.g_signs = g_signs

    g_idx_i = torch_inverse_3x3_precompute.g_idx_i
    g_idx_j = torch_inverse_3x3_precompute.g_idx_j
    g_signs = torch_inverse_3x3_precompute.g_signs

    B, N, _, _ = R.shape

    minors = R.new_zeros(B, N, 3, 3, 2, 2)
    idx_i = g_idx_i.to(R.device, non_blocking=True)  # almost never need to copy
    idx_j = g_idx_j.to(R.device, non_blocking=True)  # almost never need to copy
    signs = g_signs.to(R.device, non_blocking=True)  # almost never need to copy

    for i in range(3):
        for j in range(3):
            minors[:, :, i, j, :, :] = R[:, :, idx_i[i, j], idx_j[i, j]]

    minors = minors[:, :, :, :, 0, 0] * minors[:, :, :, :, 1, 1] - minors[:, :, :, :, 0, 1] * minors[:, :, :, :, 1, 0]
    cofactors = minors * signs[None, None]  # 3,3 -> B,N,3,3
    cofactors_t = cofactors.transpose(-2, -1)  # B, N, 3, 3
    determinant = R[:, :, 0, 0] * minors[:, :, 0, 0] - R[:, :, 0, 1] * minors[:, :, 0, 1] + R[:, :, 0, 2] * minors[:, :, 0, 2]  # B, N
    inverse = cofactors_t / (determinant[:, :, None, None] + eps)

    return inverse


@torch.jit.script
def torch_inverse_3x3(R: torch.Tensor, eps: float = torch.finfo(torch.float).eps):
    # n_batch, n_bones, 3, 3
    """
    a, b, c | m00, m01, m02
    d, e, f | m10, m11, m12
    g, h, i | m20, m21, m22
    """

    # convenient access
    r00 = R[..., 0, 0]
    r01 = R[..., 0, 1]
    r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]
    r11 = R[..., 1, 1]
    r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]
    r21 = R[..., 2, 1]
    r22 = R[..., 2, 2]

    M = torch.empty_like(R)

    # determinant of matrix minors
    # fmt: off
    M[..., 0, 0] =   r11 * r22 - r21 * r12
    M[..., 1, 0] = - r10 * r22 + r20 * r12
    M[..., 2, 0] =   r10 * r21 - r20 * r11
    M[..., 0, 1] = - r01 * r22 + r21 * r02
    M[..., 1, 1] =   r00 * r22 - r20 * r02
    M[..., 2, 1] = - r00 * r21 + r20 * r01
    M[..., 0, 2] =   r01 * r12 - r11 * r02
    M[..., 1, 2] = - r00 * r12 + r10 * r02
    M[..., 2, 2] =   r00 * r11 - r10 * r01
    # fmt: on

    # determinant of matrix
    D = r00 * M[..., 0, 0] + r01 * M[..., 1, 0] + r02 * M[..., 2, 0]

    # inverse of 3x3 matrix
    M = M / (D[..., None, None] + eps)

    return M


@torch.jit.script
def torch_inverse_2x2(A: torch.Tensor, eps: float = torch.finfo(torch.float).eps):
    a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    B = torch.empty_like(A)
    B[..., 0, 0], B[..., 0, 1] = d / det, -b / det
    B[..., 1, 0], B[..., 1, 1] = -c / det, a / det
    B = torch.where(det[..., None, None] != 0, B, torch.full_like(A, float('nan')))
    return B


def torch_unique_with_indices_and_inverse(x, dim=0):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    indices, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, indices.new_empty(unique.size(dim)).scatter_(dim, indices, perm), inverse
