import torch
from torch.nn import functional as F
from easyvolcap.utils.math_utils import normalize


def schlick_bias(x, s): return (s * x) / ((s - 1) * x + 1)


def schlick_gain(x, s): return torch.where(x < 0.5, schlick_bias(2 * x, s) / 2, (schlick_bias(2 * x - 1, 1 - s) + 1) / 2)


# implement the inverse distance sampling stragety of mipnerf360


def linear_sampling(*shape,
                    device,
                    perturb=False,
                    dtype=torch.float,
                    ) -> torch.Tensor:
    # Extract number of samples
    S = shape[-1]

    # Calculate the steps for each ray
    s_vals = torch.linspace(0., 1. - 1 / S, steps=S, device=device, dtype=dtype)  # S,

    # Expand to full shape
    for _ in range(len(shape) - 1): s_vals = s_vals[None]  # expand dim
    s_vals = s_vals.expand(shape)

    # No single jitter, use full jitter
    if perturb: s_vals = s_vals + torch.rand_like(s_vals) / S  # S,
    else: s_vals = s_vals + 0.5 / S  # S,
    return s_vals


def raw2outputs(raw, z_vals, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        dpt_map: [num_rays]. Estimated distance to object.
    """
    rgb = raw[..., :-1]  # [n_rays, n_samples, 3]
    alpha = raw[..., -1]  # [n_rays, n_samples]

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1), dtype=alpha.dtype, device=alpha.device), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]

    dpt_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(dpt_map, dtype=dpt_map.dtype, device=dpt_map.device),
                              dpt_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, dpt_map


def raw2alpha(raws: torch.Tensor, dists=0.005, bias=0.0):
    if isinstance(dists, torch.Tensor):
        if dists.ndim == raws.ndim - 1:
            dists = dists[..., None]
    return 1. - torch.exp(-(raws + bias) * dists)


def alpha2raw(alpha, dists=0.005, bias=0.0):
    return F.relu(-torch.log(1 - alpha) / dists) - bias


def alpha2sdf(alpha, beta, dists=0.005):
    return beta * torch.log(2 * beta * (-torch.log(1 - alpha) / dists))


def sdf_to_occ(sdf: torch.Tensor, beta: torch.Tensor, dists=0.005):
    sigma = sdf_to_sigma(sdf, beta)
    occ = raw2alpha(sigma, dists)
    return occ


# @torch.jit.script  # will fuse element wise operations together to make a faster invokation
def compute_val0(x: torch.Tensor, beta: torch.Tensor, ind0: torch.Tensor):
    # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
    val0 = 1 / beta * (0.5 * (x * ind0 / beta).exp()) * ind0
    return val0


# @torch.jit.script  # will fuse element wise operations together to make a faster invokation
def compute_val1(x: torch.Tensor, beta: torch.Tensor, ind1: torch.Tensor):
    # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
    val1 = 1 / beta * (1 - 0.5 * (-x * ind1 / beta).exp()) * ind1
    return val1


def sdf_to_sigma(sdf: torch.Tensor, beta: torch.Tensor):
    # double the computation, but no synchronization needed
    x = -sdf
    ind0 = x <= 0
    ind1 = ~ind0

    return compute_val0(x, beta, ind0) + compute_val1(x, beta, ind1)


def sigma_to_alpha(raw, dists=0.005, act_fn=F.softplus): return 1. - torch.exp(-act_fn(raw) * dists)


@torch.jit.script
def compute_dist(z_vals: torch.Tensor, dist_default: float = -1.0):
    if dist_default < 0:
        dist = z_vals[..., 1:] - z_vals[..., :-1]  # (B, P, S - 1)
        dist = torch.cat([dist, dist[..., -1:]], dim=-1)  # (B, P, S)
    else:
        dist = torch.full_like(z_vals, dist_default)  # (B, P, S)
    dist = dist.clip(0)  # sometimes sampled points are sooo close on a ray that the computed dists is ever so slightly negative
    return dist


@torch.jit.script
def ray2xyz(ray_o: torch.Tensor, ray_d: torch.Tensor, t: torch.Tensor, z_vals: torch.Tensor, dist_default: float = -1.0):
    # ray_o: B, P, 3
    # ray_d: B, P, 3
    # z_vals: B, P, S
    # batch: dotdict

    B, P, S = z_vals.shape[:3]
    if z_vals.ndim > ray_o.ndim:
        z_vals = z_vals[..., 0]

    # convert: B, P, S, 3
    xyz = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., None]  # B, P, S, 3
    dir = ray_d[..., None, :].expand(B, P, S, 3)  # B, P, S, 3
    dir = normalize(dir)  # (B, P, S, 3), always normalize ray direction
    t = t[..., None, :].expand(B, P, S, 1)

    # convert: B, P*S, 3
    xyz = xyz.reshape(B, P * S, 3)
    dir = dir.reshape(B, P * S, 3)
    t = t.reshape(B, P * S, 1)
    dist = compute_dist(z_vals, dist_default).reshape(B, P * S, 1)

    return xyz, dir, t, dist


def cumprod(value: torch.Tensor, dim: int = -1):
    """Custom implementation of cumprod without backward synchronization
       Just live with it, cuz this custom implementation is slower in forwarding

    Args:
        value (torch.Tensor): The value to be comproded
        dim (int, optional): The dimesion to be cromproded. Defaults to -1.

    Returns:
        torch.Tensor: The cummulative product
    """
    sh = value.shape
    dim = len(sh[:dim])  # make dim a non-negative number (i.e. -2 to 1?)
    n = value.shape[dim]  # size of dim
    out = [value.new_ones(value.shape[:dim] + value.shape[dim:]) for _ in range(n)]
    out[0] = value[(slice(None),) * dim + (slice(0, 1), )]  # store the first value
    for i in range(1, n):
        out[i] = value[(slice(None),) * dim + (slice(i, i + 1), )] * out[i - 1]
    out = torch.cat(out, dim=dim)
    return out


def render_weights_trans(occ: torch.Tensor, return_trans=False):
    # TODO: implement this as cumsum before exp operation
    # ?: https://github.com/sarafridov/K-Planes/blob/main/plenoxels/raymarching/ray_samplers.py
    # occ: n_batch, n_rays, n_samples
    if occ.ndim == 4: occ = occ[..., -1]  # get last dim
    start = occ.new_ones(occ.shape[:-1] + (1, ))
    trans = 1 - occ  # this kernel is strange...
    trans = torch.cat([start, trans], dim=-1)

    # # Choose the implementation
    # local_cumprod = torch.cumprod  # 10 more fps (73-83 fps)
    # if occ.requires_grad: local_cumprod = cumprod

    # Performing the actual integration
    trans = torch.cumprod(trans, dim=-1)[..., :-1]
    weights = occ * trans  # (n_batch, n_rays, n_samples)
    if return_trans: return weights, trans
    else: return weights


@torch.jit.script
def render_weights(occ: torch.Tensor):
    # TODO: implement this as cumsum before exp operation
    # ?: https://github.com/sarafridov/K-Planes/blob/main/plenoxels/raymarching/ray_samplers.py
    # occ: n_batch, n_rays, n_samples
    # FIXME: Assumption: greater than one sample
    if occ.shape[-1] == 1 and occ.ndim > 3: occ = occ[..., -1]  # get last dim
    start = occ.new_ones(occ.shape[:-1] + (1, ))
    trans = 1 - occ  # this kernel is strange...
    trans = torch.cat([start, trans], dim=-1)

    # # Choose the implementation
    # local_cumprod = torch.cumprod  # 10 more fps (73-83 fps)
    # if occ.requires_grad: local_cumprod = cumprod

    # Performing the actual integration
    trans = torch.cumprod(trans, dim=-1)[..., :-1]
    weights = occ * trans  # (n_batch, n_rays, n_samples)
    return weights


@torch.jit.script
def render_rgb_acc(weights: torch.Tensor, rgb: torch.Tensor):
    if weights.ndim == rgb.ndim - 1: weights = weights[..., None]
    rgb_map = torch.sum(weights * rgb, dim=-2)  # (n_batch, n_rays, 3)
    acc_map = torch.sum(weights, dim=-2)  # (n_batch, n_rays, 1)
    return rgb_map, acc_map


def volume_rendering(rgb: torch.Tensor, occ: torch.Tensor, bg_brightness: float = 0.0):
    # NOTE: here occ's last dim is not 1, but n_samples
    # rgb: n_batch, n_rays, n_samples, 3
    # occ: n_batch, n_rays, n_samples, 1
    # bg_image: n_batch, n_rays, 3 or None, if this is given as not None, the last sample on the ray will be replaced by this value (assuming this lies on the background)
    # We need to assume:
    # 1. network will find the True geometry, thus giving the background image its real value
    # 2. background image is rendered in a non-static fasion
    # returns:
    # weights: n_batch, n_rays, n_samples
    # rgb_map: n_batch, n_rays, 3
    # acc_map: n_batch, n_rays, 1

    weights = render_weights(occ)  # (n_batch, n_rays, n_samples)
    rgb_map, acc_map = render_rgb_acc(weights, rgb)
    rgb_map = rgb_map + (1. - acc_map) * bg_brightness

    return weights, rgb_map, acc_map
