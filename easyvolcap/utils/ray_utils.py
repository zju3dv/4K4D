import torch
from torch import nn
from easyvolcap.utils.math_utils import torch_inverse_3x3, normalize


@torch.jit.script
def create_meshgrid(H: int, W: int, device: torch.device = torch.device('cuda'), indexing: str = 'ij', ndc: bool = False,
                    correct_pix: bool = True, dtype: torch.dtype = torch.float):
    # kornia has meshgrid, but not the best
    i = torch.arange(H, device=device, dtype=dtype)
    j = torch.arange(W, device=device, dtype=dtype)
    if correct_pix:
        i = i + 0.5
        j = j + 0.5
    if ndc:
        i = i / H * 2 - 1
        j = j / W * 2 - 1
    ij = torch.meshgrid(i, j, indexing=indexing)  # defaults to ij
    ij = torch.stack(ij, dim=-1)  # Ht, Wt, 2

    return ij


def get_rays(H: int, W: int, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor, is_inv_K: bool = False,
             z_depth: bool = False, correct_pix: bool = True, ret_coord: bool = False):
    # calculate the world coodinates of pixels
    i, j = torch.meshgrid(torch.arange(H, dtype=R.dtype, device=R.device),
                          torch.arange(W, dtype=R.dtype, device=R.device),
                          indexing='ij')
    bss = K.shape[:-2]
    for _ in range(len(bss)): i, j = i[None], j[None]
    i, j = i.expand(bss + i.shape[len(bss):]), j.expand(bss + j.shape[len(bss):])
    # 0->H, 0->W
    return get_rays_from_ij(i, j, K, R, T, is_inv_K, z_depth, correct_pix, ret_coord)


def get_rays_from_ij(i: torch.Tensor, j: torch.Tensor,
                     K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                     is_inv_K: bool = False, use_z_depth: bool = False,
                     correct_pix: bool = True, ret_coord: bool = False):
    # i: B, P or B, H, W or P or H, W
    # j: B, P or B, H, W or P or H, W
    # K: B, 3, 3
    # R: B, 3, 3
    # T: B, 3, 1
    nb_dim = len(K.shape[:-2])  # number of batch dimensions
    np_dim = len(i.shape[nb_dim:])  # number of points dimensions
    if not is_inv_K: invK = torch_inverse_3x3(K.float()).type(K.dtype)
    else: invK = K
    ray_o = - R.mT @ T  # B, 3, 1

    # Prepare the shapes
    for _ in range(np_dim): invK = invK.unsqueeze(-3)
    invK = invK.expand(i.shape + (3, 3))
    for _ in range(np_dim): R = R.unsqueeze(-3)
    R = R.expand(i.shape + (3, 3))
    for _ in range(np_dim): T = T.unsqueeze(-3)
    T = T.expand(i.shape + (3, 1))
    for _ in range(np_dim): ray_o = ray_o.unsqueeze(-3)
    ray_o = ray_o.expand(i.shape + (3, 1))

    # Pixel center correction
    if correct_pix: i, j = i + 0.5, j + 0.5
    else: i, j = i.float(), j.float()

    # 0->H, 0->W
    # int -> float; # B, H, W, 3, 1 or B, P, 3, 1 or P, 3, 1 or H, W, 3, 1
    xy1 = torch.stack([j, i, torch.ones_like(i)], dim=-1)[..., None]
    pixel_camera = invK @ xy1  # B, H, W, 3, 1 or B, P, 3, 1
    pixel_world = R.mT @ (pixel_camera - T)  # B, P, 3, 1

    # Calculate the ray direction
    pixel_world = pixel_world[..., 0]
    ray_o = ray_o[..., 0]
    ray_d = pixel_world - ray_o  # use pixel_world depth as is (no curving)
    if not use_z_depth: ray_d = normalize(ray_d)  # B, P, 3, 1

    if not ret_coord: return ray_o, ray_d
    elif correct_pix: return ray_o, ray_d, (torch.stack([i, j], dim=-1) - 0.5).long()  # B, P, 2
    else: return ray_o, ray_d, torch.stack([i, j], dim=-1).long()  # B, P, 2


def weighted_sample_rays(wet: torch.Tensor,  # weight of every pixel (high weights -> sample more)
                         K: torch.Tensor,  # intrinsic
                         R: torch.Tensor,  # extrinsic
                         T: torch.Tensor,  # extrinsic
                         n_rays: int = -1,  # -1 means use all pixels (non-zero mask)
                         use_z_depth: bool = False,
                         correct_pix: bool = True,
                         ):
    # When random is set, will ignore n_rays and sample the whole image
    # 1. Need to consider incorporating masks, bounds (only some of the pixels are sampled)
    # 2. Exactly how slow can the vanilla implementation be?
    # with timeit(weighted_sample_coords.__name__):
    coords = weighted_sample_coords(wet, n_rays)  # B, P, 2
    i, j = coords.unbind(-1)

    # Maybe refactor these? Built rays directly
    # with timeit(get_rays_from_ij.__name__):
    ray_o, ray_d = get_rays_from_ij(i, j, K, R, T, use_z_depth=use_z_depth, correct_pix=correct_pix)  # MARK: 0.xms
    return ray_o, ray_d, coords  # gt, input, input, others


def indices_to_coords(idx: torch.Tensor, H: int, W: int):
    i = idx // W
    j = idx % W
    return torch.stack([i, j], dim=-1)  # ..., 2


def coords_to_indices(ij: torch.Tensor, H, W):
    i, j = ij.unbind(-1)
    return i * W + j  # ... (no last dim)


def weighted_sample_coords(wet: torch.Tensor, n_rays: int = -1) -> torch.Tensor:
    # Here, msk is a sampling weight controller, and only those with >0 values will be sampled

    # Non training sampling
    if n_rays == -1: return torch.nonzero(torch.ones_like(wet[..., 0]))  # MARK: assume sorted

    sh = wet.shape
    H, W = wet.shape[-3:-1]
    weight = wet.view(*sh[:-3], -1)  # B, H, W, 1 ->  B, -1 (H * W)
    if weight.sum() == weight.numel():  # ?: Will this be too slow?
        indices = torch.randint(0, H * W, (n_rays,), device=wet.device)  # MARK: 0.1ms
    else:
        indices = torch.multinomial(weight, n_rays, replacement=True)  # B, P, # MARK: 3-5ms
    coords = indices_to_coords(indices, H, W)  # B, P, 2 # MARK: 0.xms
    return coords


def parameterize(x: torch.Tensor, v: torch.Tensor):
    s = (-x * v).sum(dim=-1, keepdim=True)  # closest distance along ray to origin
    o = s * v + x  # closest point along ray to origin
    return o, s
