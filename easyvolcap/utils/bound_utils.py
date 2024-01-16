import torch


@torch.jit.script
def monotonic_near_far(near: torch.Tensor, far: torch.Tensor, n: torch.Tensor, f: torch.Tensor):
    n = n[..., None, None]
    f = f[..., None, None]
    near, far = near.clip(n, f), far.clip(n, f)
    valid_mask = near < far
    valid_near_plane = torch.where(valid_mask, near, f).min()
    valid_far_plane = torch.where(valid_mask, far, n).max()
    near, far = torch.where(valid_mask, near, valid_near_plane), torch.where(valid_mask, far, valid_far_plane)  # what ever for these points
    near, far = near.clip(n, f), far.clip(n, f)
    return near, far


def get_bound_corners(bounds: torch.Tensor):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = bounds.new_tensor([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


@torch.jit.script
def project(xyz: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor):
    """
    xyz: [...N, 3], ... means some batch dim
    K: [3, 3]
    R: [3, 3]
    T: [3, 1]
    """
    RT = torch.cat([R, T], dim=-1)
    xyz = xyz @ RT[..., :3].mT + RT[..., 3:].mT
    xyz = xyz @ K.mT
    xy = xyz[..., :2] / xyz[..., 2:]
    return xy


@torch.jit.script
def transform(xyz, RT):
    """
    xyz: [...N, 3], ... means some batch dim
    RT: [3, 4]
    """
    xyz = xyz @ RT[:, :3].T + RT[:, 3:].T
    return xyz


def get_bound_2d_bound(bounds: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor, H, W, pad=25):  # pad more, be safe
    if bounds.ndim == 3: corners_3d = torch.stack([get_bound_corners(b) for b in bounds])
    else: corners_3d = get_bound_corners(bounds)
    if isinstance(H, torch.Tensor): H = H.item()
    if isinstance(W, torch.Tensor): W = W.item()
    corners_2d = project(corners_3d, K, R, T)
    corners_2d = corners_2d.round().int()

    x_min = (corners_2d[..., 0].min() - pad).clip(0, W)
    x_max = (corners_2d[..., 0].max() + pad).clip(0, W)
    y_min = (corners_2d[..., 1].min() - pad).clip(0, H)
    y_max = (corners_2d[..., 1].max() + pad).clip(0, H)
    x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

    return x, y, w, h


def get_bound_3d_near_far(bounds: torch.Tensor, R: torch.Tensor, T: torch.Tensor):
    corners_3d_worlds = get_bound_corners(bounds)
    corners_3d_camera = transform(corners_3d_worlds, torch.cat([R, T], dim=-1))
    near = corners_3d_camera[..., -1].min()
    far = corners_3d_camera[..., -1].max()
    return near, far


# MipNeRF360 space contraction


@torch.jit.script
def contract(x: torch.Tensor, r: float = 1.0, p: float = torch.inf):
    l = x.norm(dim=-1, keepdim=True, p=p) + 1e-13
    m = l <= r

    # For smaller than radius points: x = x
    # For larger than radius points: (2 - r/|x|) * r * x / |x|
    x = x * m + ~m * (2 - r / l) * r * x / l
    return x


def get_bounds(xyz, padding=0.05):  # 5mm padding? really?
    # xyz: n_batch, n_points, 3

    min_xyz = torch.min(xyz, dim=1)[0]  # torch min with dim is ...
    max_xyz = torch.max(xyz, dim=1)[0]
    min_xyz -= padding
    max_xyz += padding
    bounds = torch.stack([min_xyz, max_xyz], dim=1)
    return bounds
    diagonal = bounds[..., 1:] - bounds[..., :1]  # n_batch, 1, 3
    bounds[..., 1:] = bounds[..., :1] + torch.ceil(diagonal / voxel_size) * voxel_size  # n_batch, 1, 3
    return bounds


@torch.jit.script
def get_near_far_aabb(bounds: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor, epsilon: float = 1e-8):
    """
    calculate intersections with 3d bounding box
    bounds: n_batch, 2, 3, min corner and max corner
    ray_o: n_batch, n_points, 3
    ray_d: n_batch, n_points, 3, assume already normalized
    return: near, far: B, P, 1

    NOTE: This function might produce inf or -inf, need a clipping
    """
    if ray_o.ndim >= bounds.ndim:
        diff = ray_o.ndim - bounds.ndim
        for i in range(diff):
            bounds = bounds.unsqueeze(-3)  # match the batch dimensions, starting from second

    # NOTE: here, min in tmin means the intersection with point bound_min, not minimum
    tmin = (bounds[..., :1, :] - ray_o) / (ray_d + epsilon)  # (b, 1, 3) - (b, 1, 3) / (b, n, 3) -> (b, n, 3)
    tmax = (bounds[..., 1:, :] - ray_o) / (ray_d + epsilon)  # (b, n, 3)
    # near plane is where the intersection has a smaller value on corresponding dimension than the other point
    t1 = torch.minimum(tmin, tmax)  # (b, n, 3)
    t2 = torch.maximum(tmin, tmax)
    # near plane is the maximum of x, y, z intersection point, entering AABB: enter every dimension
    near: torch.Tensor = t1.max(dim=-1, keepdim=True)[0]  # (b, n)
    far: torch.Tensor = t2.min(dim=-1, keepdim=True)[0]
    return near, far


def sample_depth_near_far(near: torch.Tensor, far: torch.Tensor, n_samples: int, perturb: bool = False):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    s_vals = torch.linspace(0., 1., steps=n_samples, dtype=near.dtype, device=near.device)
    z_vals = near * (1. - s_vals) + far * s_vals

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        z_rand = torch.rand(*z_vals.shape, dtype=upper.dtype, device=upper.device)
        z_vals = lower + (upper - lower) * z_rand

    return z_vals


def sample_points_near_far(ray_o, ray_d, near, far, n_samples: int, perturb: bool):
    # ray_o: n_batch, n_rays, 3
    # ray_d: n_batch, n_rays, 3
    # near: n_batch, n_rays
    # far: n_batch, n_rays
    # sample points and depth values for every ray between near-far
    # and possibly do some pertubation to introduce randomness in training
    # dists: n_batch, n_rays, n_steps
    # pts: n_batch, n_rays, n_steps, 3

    # calculate the steps for each ray
    z_vals = sample_depth_near_far(near, far, n_samples, perturb)

    # (n_batch, n_rays, n_samples, 3)
    xyz = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., None]

    return xyz, z_vals
