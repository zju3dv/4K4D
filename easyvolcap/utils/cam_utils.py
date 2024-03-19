from __future__ import annotations
import os
import cv2
import json
import torch
import numpy as np
from typing import Union
from enum import Enum, auto

from scipy import interpolate
from scipy.spatial.transform import Rotation

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.chunk_utils import multi_gather
from easyvolcap.utils.data_utils import get_rays, get_near_far
from easyvolcap.utils.math_utils import affine_inverse, affine_padding


def compute_camera_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # c2ws = affine_inverse(w2cs)  # N, L, 3, 4
    # src_exts = affine_padding(w2cs)  # N, L, 4, 4

    # tar_c2ws = c2ws
    # src_c2ws = affine_inverse(src_exts)
    centers_target = tar_c2ws[..., :3, 3]  # N, L, 3
    centers_source = src_c2ws[..., :3, 3]  # N, L, 3

    # Using distance between centers for camera selection
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # N, N, L,

    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # similarity to source views # Target, Source, Latent
    return src_sims, src_inds  # N, N, L


def compute_camera_zigzag_similarity(tar_c2ws: torch.Tensor, src_c2ws: torch.Tensor):
    # Get the camera centers
    centers_target = tar_c2ws[..., :3, 3]  # (Vt, F, 3)
    centers_source = src_c2ws[..., :3, 3]  # (Vs, F, 3)

    # Compute the distance between the centers
    sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # (Vt, Vs, F)
    # Source view index and there similarity
    src_sims, src_inds = sims.sort(dim=1, descending=True)  # (Vt, Vs, F), (Vt, Vs, F)

    # Select the closest source view as the reference view for each target view
    ref_view = multi_gather(centers_source.permute(1, 0, 2), src_inds.permute(2, 0, 1)[..., 0]).permute(1, 0, 2)  # (Vt, F, 3)

    # Compute the cross product between the reference view and target view, and the cross product between the source views and the target view
    ref_cross = torch.cross(ref_view, centers_target, dim=-1)  # (Vt, F, 3)
    src_cross = torch.cross(centers_source[None], centers_target[:, None], dim=-1)  # (Vt, Vs, F, 3)

    # Compute the inner product between the cross products to determine the zigzag placing
    zigzag = (ref_cross[:, None] * src_cross).sum(dim=-1)  # (Vt, Vs, F)

    zigzag_src_sims, zigzag_src_inds = src_sims.clone(), src_inds.clone()
    # Re-indexing the similarity and indices
    for v in range(len(zigzag)):
        # Get the sorted zig and zag similarity and indices respectively
        zig_msk = torch.sum(torch.eq(torch.arange(len(centers_source))[zigzag[v, :, 0] > 0][:, None], src_inds[v, :, 0]), dim=0).bool()
        zig_src_sims, zig_src_inds = src_sims[v][zig_msk], src_inds[v][zig_msk]  # (L, F), (L, F)
        zag_msk = torch.sum(torch.eq(torch.arange(len(centers_source))[zigzag[v, :, 0] < 0][:, None], src_inds[v, :, 0]), dim=0).bool()
        zag_src_sims, zag_src_inds = src_sims[v][zag_msk], src_inds[v][zag_msk]  # (R, F), (R, F)

        # Concatenate the zig and zag similarity and indices in order zig-zag-zig-zag-...
        size = min(len(zig_src_sims), len(zag_src_sims))
        zigzag_src_sims[v, 0:size * 2:2], zigzag_src_sims[v, 1:size * 2:2] = zig_src_sims[:size], zag_src_sims[:size]  # (S*2, F), (S*2, F)
        zigzag_src_inds[v, 0:size * 2:2], zigzag_src_inds[v, 1:size * 2:2] = zig_src_inds[:size], zag_src_inds[:size]  # (S*2, F), (S*2, F)

        # Concatenate the remaining similarity and indices
        if len(zig_src_sims) > len(zag_src_sims): zigzag_src_sims[v, size * 2:], zigzag_src_inds[v, size * 2:] = zig_src_sims[size:], zig_src_inds[size:]
        else: zigzag_src_sims[v, size * 2:], zigzag_src_inds[v, size * 2:] = zag_src_sims[size:], zag_src_inds[size:]

    # Return the zigzag similarity and indices
    return zigzag_src_sims, zigzag_src_inds


class Sourcing(Enum):
    # Type of source indexing
    DISTANCE = auto()  # the default source indexing
    ZIGZAG = auto()  # will index the source view in zigzag order


class Interpolation(Enum):
    # Type of interpolation to use
    CUBIC = auto()  # the default interpolation
    ORBIT = auto()  # will find a full circle around the cameras, the default orbit path
    SPIRAL = auto()  # will perform spiral motion around the cameras
    SECTOR = auto()  # will find a circular sector around the cameras
    NONE = auto()  # used as is


def normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-13)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


# From https://github.com/NVLabs/instant-ngp


def compute_center_of_attention(c2ws: np.ndarray):
    # TODO: Should vectorize this to make it faster, this is not very tom94
    totw = 0.0
    totp = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    for mf in c2ws:
        for mg in c2ws:
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    return totp[..., None]  # 3, 1


def closest_point_2_lines(oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray):
    # Returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    return (oa + ta * da + ob + tb * db) * 0.5, denom


# From: https://github.com/sarafridov/K-Planes/blob/main/plenoxels/datasets/ray_utils.py


def average_c2ws(c2ws: np.ndarray, align_cameras: bool = True, look_at_center: bool = True) -> np.ndarray:
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """

    if align_cameras:
        # 1. Compute the center
        center = compute_center_of_attention(c2ws)[..., 0]  # (3)
        # 2. Compute the z axis
        z = -normalize(c2ws[..., 1].mean(0))  # (3) # FIXME: WHY?
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 2].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    else:
        # 1. Compute the center
        center = c2ws[..., 3].mean(0)  # (3)
        # 2. Compute the z axis
        if look_at_center:
            look = compute_center_of_attention(c2ws)[..., 0]  # (3)
            z = normalize(look - center)
        else:
            z = normalize(c2ws[..., 2].mean(0))  # (3)
        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = c2ws[..., 1].mean(0)  # (3)
        # 4. Compute the x axis
        x = -normalize(np.cross(z, y_))  # (3)
        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = -np.cross(x, z)  # (3)

    c2w_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return c2w_avg


def align_c2ws(c2ws: np.ndarray, c2w_avg: Union[np.ndarray, None] = None) -> np.ndarray:
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    c2w_avg = c2w_avg if c2w_avg is not None else average_c2ws(c2ws)  # (3, 4)
    c2w_avg_homo = np.eye(4, dtype=c2ws.dtype)
    c2w_avg_homo[:3] = c2w_avg  # convert to homogeneous coordinate for faster computation

    last_row = np.tile(np.asarray([0, 0, 0, 1], dtype=np.float32), (len(c2ws), 1, 1))  # (N_images, 1, 4)
    c2ws_homo = np.concatenate([c2ws, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    c2ws_centered = np.linalg.inv(c2w_avg_homo) @ c2ws_homo  # (N_images, 4, 4)
    c2ws_centered = c2ws_centered[:, :3]  # (N_images, 3, 4)

    return c2ws_centered


def average_w2cs(w2cs: np.ndarray) -> np.ndarray:
    # Transform the world2camera extrinsic from matrix representation to vector representation
    rvecs = np.array([cv2.Rodrigues(w2c[:3, :3])[0] for w2c in w2cs], dtype=np.float32)  # (V, 3, 1)
    tvecs = w2cs[:, :3, 3:]  # (V, 3, 1)

    # Compute the average view direction and center in vector mode
    rvec_avg = rvecs.mean(axis=0)  # (3, 1)
    tvec_avg = tvecs.mean(axis=0)  # (3, 1)

    # Back to matrix representation
    w2c_avg = np.concatenate([cv2.Rodrigues(rvec_avg)[0], tvec_avg], axis=1)
    return w2c_avg


def gen_cam_interp_func_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    center_t, center_u, front_t, front_u, up_t, up_u = gen_cam_interp_params_bspline(c2ws, smoothing_term, per)

    def f(us: np.ndarray):
        if isinstance(us, int) or isinstance(us, float): us = np.asarray([us])
        if isinstance(us, list): us = np.asarray(us)

        # The interpolation t
        center = np.asarray(interpolate.splev(us, center_t)).T.astype(c2ws.dtype)
        v_front = np.asarray(interpolate.splev(us, front_t)).T.astype(c2ws.dtype)
        v_up = np.asarray(interpolate.splev(us, up_t)).T.astype(c2ws.dtype)

        # Normalization
        v_front = normalize(v_front)
        v_up = normalize(v_up)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)

        # Combination
        render_c2ws = np.stack([v_right, v_down, v_front, center], axis=-1)
        return render_c2ws
    return f


def gen_cam_interp_params_bspline(c2ws: np.ndarray, smoothing_term=1.0, per: int = 0):
    """Return B-spline interpolation parameters for the camera # MARK: Quite easy to error out
    Actually this should be implemented as a general interpolation function
    Reference get_camera_up_front_center for the definition of worldup, front, center
    Args:
        smoothing_term(float): degree of smoothing to apply on the camera path interpolation
    """
    centers = c2ws[..., :3, 3]
    fronts = c2ws[..., :3, 2]
    ups = -c2ws[..., :3, 1]

    center_t, center_u = interpolate.splprep(centers.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    front_t, front_u = interpolate.splprep(fronts.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    up_t, up_u = interpolate.splprep(ups.T, s=smoothing_term, per=per)  # array of u corresponds to parameters of specific camera points
    return center_t, center_u, front_t, front_u, up_t, up_u


def cubic_spline(us: np.ndarray, N: int):
    if isinstance(us, int) or isinstance(us, float): us = np.asarray([us])
    if isinstance(us, list): us = np.asarray(us)

    # Preparation
    t = (N - 1) * us  # expanded to the length of the sequence
    i0 = np.floor(t).astype(np.int32) - 1
    i0 = np.where(us != 1.0, i0, i0 - 1)  # remove end point (nans for 1s)
    i1 = i0 + 1
    i2 = i0 + 2
    i3 = i0 + 3
    i0, i1, i2, i3 = np.clip(i0, 0, N - 1), np.clip(i1, 0, N - 1), np.clip(i2, 0, N - 1), np.clip(i3, 0, N - 1)
    t0, t1, t2, t3 = i0 / (N - 1), i1 / (N - 1), i2 / (N - 1), i3 / (N - 1)
    t = (t - i1)  # normalize to the start?
    t = t.astype(np.float32)  # avoid fp64 problems

    # Compute coeffs
    tt = t * t
    ttt = tt * t
    a = (1 - t) * (1 - t) * (1 - t) * (1. / 6.)
    b = (3. * ttt - 6. * tt + 4.) * (1. / 6.)
    c = (-3. * ttt + 3. * tt + 3. * t + 1.) * (1. / 6.)
    d = ttt * (1. / 6.)

    t0, t1, t2, t3 = t0.astype(np.float32), t1.astype(np.float32), t2.astype(np.float32), t3.astype(np.float32)
    a, b, c, d = a.astype(np.float32), b.astype(np.float32), c.astype(np.float32), d.astype(np.float32)

    return t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d)


class InterpolatingExtrinsics:
    def __init__(self, c2w: np.ndarray) -> None:
        self.Q = Rotation.from_matrix(c2w[..., :3, :3]).as_quat()
        self.T = c2w[..., :3, 3]

    def __add__(lhs, rhs: InterpolatingExtrinsics):  # FIXME: Dangerous
        Ql, Qr = lhs.Q, rhs.Q
        Qr = np.where((Ql * Qr).sum(axis=-1, keepdims=True) < 0, -Qr, Qr)
        lhs.Q = Ql + Qr
        lhs.T = lhs.T + rhs.T
        return lhs

    def __radd__(rhs, lhs: InterpolatingExtrinsics):
        return rhs.__add__(lhs)

    def __mul__(lhs, rhs: np.ndarray):
        lhs.Q = rhs[..., None] * lhs.Q
        lhs.T = rhs[..., None] * lhs.T
        return lhs  # inplace modification

    def __rmul__(rhs, lhs: np.ndarray):
        return rhs.__mul__(lhs)

    def numpy(self):
        return np.concatenate([Rotation.from_quat(self.Q).as_matrix(), self.T[..., None]], axis=-1).astype(np.float32)


def gen_cubic_spline_interp_func(c2ws: np.ndarray, smoothing_term=10.0, *args, **kwargs):
    # Split interpolation
    N = len(c2ws)
    assert N > 3, 'Cubic Spline interpolation requires at least four inputs'
    if smoothing_term == 0:
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = N - 1 + 4 - 2  # should remove last one segment, please just work...
        c2ws = np.concatenate([c2ws[-2:], c2ws, c2ws[:2]])

    def lf(us: np.ndarray):
        N = len(c2ws)  # should this be recomputed?
        t, (i0, i1, i2, i3), (t0, t1, t2, t3), (a, b, c, d) = cubic_spline(us, N)

        # Extra inter target
        c0, c1, c2, c3 = InterpolatingExtrinsics(c2ws[i0]), InterpolatingExtrinsics(c2ws[i1]), InterpolatingExtrinsics(c2ws[i2]), InterpolatingExtrinsics(c2ws[i3])
        c = c0 * a + c1 * b + c2 * c + c3 * d  # to utilize operator overloading
        c = c.numpy()  # from InterpExt to numpy
        if isinstance(us, int) or isinstance(us, float): c = c[0]  # remove extra dim
        return c

    if smoothing_term == 0:
        def pf(us): return lf((us * N - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f


def gen_linear_interp_func(lins: np.ndarray, smoothing_term=10.0):  # smoothing_term <= will loop the interpolation
    if smoothing_term == 0:
        n = len(lins)
        low = -2  # when we view index as from 0 to n, should remove first two segments
        high = n - 1 + 4 - 2  # should remove last one segment, please just work...
        lins = np.concatenate([lins[-2:], lins, lins[:2]])

    lf = interpolate.interp1d(np.linspace(0, 1, len(lins), dtype=np.float32), lins, axis=-2)  # repeat

    if smoothing_term == 0:
        def pf(us): return lf((us * n - low) / (high - low))  # periodic function will call the linear function
        f = pf  # periodic function
    else:
        f = lf  # linear function
    return f


def interpolate_camera_path(c2ws: np.ndarray, n_render_views=50, smoothing_term=10.0, **kwargs):
    # Store interpolation parameters
    f = gen_cubic_spline_interp_func(c2ws, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=c2ws.dtype)
    return f(us)


def interpolate_camera_lins(lins: np.ndarray, n_render_views=50, smoothing_term=10.0, **kwargs):
    # Store interpolation parameters
    f = gen_linear_interp_func(lins, smoothing_term)

    # The interpolation t
    us = np.linspace(0, 1, n_render_views, dtype=lins.dtype)
    return f(us)


def generate_spiral_path(c2ws: np.ndarray,
                         n_render_views=300,
                         n_rots=2,
                         zrate=0.5,
                         percentile=70,

                         focal_offset=0.0,
                         radius_ratio=1.0,
                         xyz_ratio=[1.0, 1.0, 0.25],
                         xyz_offset=[0.0, 0.0, 0.0],
                         **kwargs) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering.
    From https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    and https://github.com/apchenstu/TensoRF/blob/main/dataLoader/llff.py
    """
    # Prepare input data
    c2ws = c2ws[..., :3, :4]

    # Center pose
    c2w_avg = average_c2ws(c2ws, align_cameras=False, look_at_center=True)  # [3, 4]

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset as a weighted average
    # of near and far bounds in disparity space.
    focal = focal_offset + np.linalg.norm(compute_center_of_attention(c2ws)[..., 0] - c2w_avg[..., 3])  # (3)

    # Get radii for spiral path using 70th percentile of camera origins.
    radii = np.percentile(np.abs(c2ws[:, :3, 3] - c2w_avg[..., 3]), percentile, 0) * radius_ratio  # N, 3
    radii = np.concatenate([xyz_ratio * radii, [1.]])  # 4,

    # Generate c2ws for spiral path.
    render_c2ws = []
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_render_views, endpoint=False):
        t = radii * [np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.] + \
            np.concatenate([xyz_offset, [0.]])

        center = c2w_avg @ t
        center = center.astype(c2ws.dtype)
        lookat = c2w_avg @ np.array([0, 0, focal, 1.0], dtype=c2ws.dtype)

        v_front = -normalize(center - lookat)
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws


def generate_hemispherical_orbit(c2ws: np.ndarray,
                                 n_render_views=50,
                                 orbit_height=0.,
                                 orbit_radius=-1,
                                 radius_ratio=1.0,
                                 **kwargs):
    """Calculates a render path which orbits around the z-axis.
    Based on https://github.com/google-research/google-research/blob/342bfc150ef1155c5254c1e6bd0c912893273e8d/regnerf/internal/datasets.py
    TODO: Implement this for non-centered camera paths
    """
    # Center pose
    c2w_avg = average_c2ws(c2ws)  # [3, 4]

    # Find the origin and radius for the orbit
    origins = c2ws[:, :3, 3]
    radius = (np.sqrt(np.mean(np.sum(origins ** 2, axis=-1))) * radius_ratio) if orbit_radius <= 0 else orbit_radius

    # Get average pose
    v_up = -normalize(c2ws[:, :3, 1].sum(0))

    # Assume that z-axis points up towards approximate camera hemispherical
    sin_phi = np.mean(origins[:, 2], axis=0) / radius
    cos_phi = np.sqrt(1 - sin_phi ** 2)
    render_c2ws = []

    for theta in np.linspace(0., 2. * np.pi, n_render_views, endpoint=False, dtype=c2ws.dtype):
        center = radius * np.asarray([cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi], dtype=c2ws.dtype)
        center[2] += orbit_height
        v_front = -normalize(center)
        center += c2w_avg[..., :3, -1]  # last dim, center of avg
        v_right = normalize(np.cross(v_front, v_up))
        v_down = np.cross(v_front, v_right)
        c2w = np.stack([v_right, v_down, v_front, center], axis=-1)  # 3, 4
        render_c2ws.append(c2w)

    render_c2ws = np.stack(render_c2ws, axis=0)  # N, 3, 4
    return render_c2ws
