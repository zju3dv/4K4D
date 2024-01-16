import torch
import torch.nn.functional as F

from typing import List
from functools import lru_cache
from easyvolcap.utils.math_utils import torch_inverse_3x3


def batch_rodrigues(
    rot_vecs: torch.Tensor,  # B, N, 3
    eps: float = torch.finfo(torch.float).eps
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor BxNx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor BxNx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[:-1]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = (rot_vecs + eps).norm(p=2, dim=-1, keepdim=True)  # B, N, 3
    rot_dir = rot_vecs / angle

    cos = angle.cos()[..., None, :]
    sin = angle.sin()[..., None, :]

    # Bx1 arrays
    rx, ry, rz = rot_dir.split(1, dim=-1)
    zeros = torch.zeros(batch_size + (1,), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view(batch_size + (3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    for i in range(len(batch_size)): ident = ident[None]
    rot_mat = ident + sin * K + (1 - cos) * K @ K
    return rot_mat


def transform_mat(R: torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype=torch.float32
) -> torch.Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def apply_r(vds, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    vds = vds.view(-1, vds.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    vds = torch.bmm(Rs, vds[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    vds = vds.view(B, N, -1)
    return vds


def apply_rt(pts, se3):
    # se3: (B, N, 6), pts: (B, N, 3)
    B, N, _ = se3.shape
    se3 = se3.view(-1, se3.shape[-1])
    pts = pts.view(-1, pts.shape[-1])
    Rs = batch_rodrigues(se3[:, :3])  # get rotation matrix: (N, 3, 3)
    pts = torch.bmm(Rs, pts[:, :, None])[:, :, 0]  # batch matmul to apply rotation, and get (N, 3) back
    # TODO: retrain these...
    pts += se3[:, 3:]  # apply transformation
    pts = pts.view(B, N, -1)
    return pts


def get_aspect_bounds(bounds) -> torch.Tensor:
    # bounds: B, 2, 3
    half_edge = (bounds[:, 1:] - bounds[:, :1]) / 2  # 1, 1, 3
    half_long_edge = half_edge.max(dim=-1, keepdim=True)[0].expand(-1, -1, 3)
    middle_point = half_edge + bounds[:, :1]  # 1, 1, 3
    return torch.cat([middle_point - half_long_edge, middle_point + half_long_edge], dim=-2)


@lru_cache
def get_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    if preserve_aspect_ratio:
        bounds = get_aspect_bounds(bounds)
    n_batch = bounds.shape[0]

    # move to -1
    # scale to 1
    # scale * 2
    # move - 1

    move0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move0[:, :3, -1] = -bounds[:, :1]

    scale0 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale0[:, torch.arange(3), torch.arange(3)] = 1 / (bounds[:, 1:] - bounds[:, :1])

    scale1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    scale1[:, torch.arange(3), torch.arange(3)] = 2

    move1 = torch.eye(4, device=bounds.device)[None].expand(n_batch, -1, -1)
    move1[:, :3, -1] = -1

    M = move1.matmul(scale1.matmul(scale0.matmul(move0)))

    return M  # only scale and translation has value


@lru_cache
def get_inv_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    invM = scale_trans_inverse(M)
    return invM


@lru_cache
def get_dir_ndc_transform(bounds: torch.Tensor, preserve_aspect_ratio: bool = False) -> torch.Tensor:
    # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
    invM = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    return invM.mT


@torch.jit.script
def scale_trans_inverse(M: torch.Tensor) -> torch.Tensor:
    n_batch = M.shape[0]
    invS = 1 / M[:, torch.arange(3), torch.arange(3)]
    invT = -M[:, :3, 3:] * invS[..., None]
    invM = torch.eye(4, device=M.device)[None].expand(n_batch, -1, -1)
    invM[:, torch.arange(3), torch.arange(3)] = invS
    invM[:, :3, 3:] = invT

    return invM


def ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    # both with batch dimension
    # pts has no last dimension
    M = get_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def inv_ndc(pts, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_inv_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def dir_ndc(dir, bounds, preserve_aspect_ratio=False) -> torch.Tensor:
    M = get_dir_ndc_transform(bounds, preserve_aspect_ratio)
    R = M[:, :3, :3]
    dir = dir.matmul(R.mT)
    return dir


@lru_cache
def get_rigid_transform(poses: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    # pose: B, N, 3
    # joints: B, N, 3
    # parents: B, N
    # B, N, _ = poses.shape
    R = batch_rodrigues(poses.view(-1, 3))  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints, parents.view(-1))  # MARK: doc of this is wrong about parent
    return J, A


def get_rigid_transform_nobatch(poses: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    # pose: N, 3
    # joints: N, 3
    # parents: N
    R = batch_rodrigues(poses)  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints[None], parents)  # MARK: doc of this is wrong about parent
    J, A = J[0], A[0]  # remove batch dimension
    return J, A


# def apply_rt(xyz: torch.Tensor, rt: torch.Tensor):
#     # xyz: B, P, 3
#     # rt: B, P, 6
#     R = batch_rodrigues(rt[..., :3].view(-1, 3)).view(rt.shape[:-1] + (3, 3))  # B, P, 3, 3
#     T = rt[..., 3:]  # B, P, 3
#     return (R @ xyz[..., None])[..., 0] + T


def mat2rt(A: torch.Tensor) -> torch.Tensor:
    """calculate 6D rt representation of blend weights and bones
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    from pytorch3d import transforms
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
    # bw
    # 1. get blended transformation from bw and bones
    # 2. get quaternion from matrix
    # 3. get axis-angle from quaternion
    # 4. slice out the translation
    # 5. concatenation
    # A = blend_transform(input, batch.A)

    r = transforms.quaternion_to_axis_angle(transforms.matrix_to_quaternion(A[..., :3, :3]))  # n_batch, n_points, 3
    t = A[..., :3, 3]  # n_batch, n_points, 3, drops last dimension
    rt = torch.cat([r, t], dim=-1)
    return rt


def screw2rt(screw: torch.Tensor) -> torch.Tensor:
    from pytorch3d import transforms
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
    return mat2rt(transforms.se3_exp_map(screw.view(-1, screw.shape[-1])).permute(0, 2, 1)).view(*screw.shape)


def blend_transform(bw: torch.Tensor, A: torch.Tensor):
    """blend the transformation
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A = (bw.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(-4)).sum(dim=-3)
    return A


def tpose_points_to_ndc_points(pts: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    R = M[:, :3, :3]
    T = M[:, :3, 3:]
    pts = pts.matmul(R.mT) + T.mT
    return pts


def tpose_dirs_to_ndc_dirs(dirs: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    R = M[:, :3, :3]
    dirs = dirs.matmul(R.mT)
    return dirs


def world_dirs_to_pose_dirs(wdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, R)
    return pts


def pose_dirs_to_world_dirs(pdirs, R):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    """
    pts = torch.matmul(pdirs, R.transpose(1, 2))
    return pts


def world_points_to_pose_points(wpts, R, Th):
    """
    wpts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :]  # add fake point dimension
    pts = torch.matmul(wpts - Th, R)
    return pts


def pose_points_to_world_points(ppts, R, Th):
    """
    ppts: n_batch, n_points, 3
    R: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    if Th.ndim == 2:
        Th = Th[..., None, :]  # add fake point dimension
    pts = torch.matmul(ppts, R.transpose(1, 2)) + Th
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R = A_bw[..., :3, :3]  # never None
    R_transpose = R.mT  # inverse transpose of inverse(R)
    pts = torch.sum(R_transpose * ddirs.unsqueeze(-2), dim=-1)
    return pts


def pose_points_to_tpose_points(ppts: torch.Tensor, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    pts = ppts - A_bw[..., :3, 3]
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    pts = torch.sum(R_inv * pts.unsqueeze(-2), dim=-1)
    return pts


def tpose_points_to_pose_points(pts, bw=None, A=None, A_bw=None, R_inv=None):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw
    R = A_bw[..., :3, :3]
    pts = torch.sum(R * pts.unsqueeze(-2), dim=-1)
    pts = pts + A_bw[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw=None, A=None, A_bw=None, R_inv=None):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, n_points, n_bones
    A: n_batch, n_bones, 4, 4
    """
    A_bw = blend_transform(bw, A) if A_bw is None else A_bw

    # since the blended rotation matrix is not a pure rotation anymore, we transform with inverse transpose
    R_inv = torch_inverse_3x3(A_bw[..., :3, :3]) if R_inv is None else R_inv
    R_inv_trans = R_inv.mT  # inverse transpose of the rotation

    pts = torch.sum(R_inv_trans * ddirs.unsqueeze(-2), dim=-1)
    return pts


world_points_to_view_points = world_points_to_pose_points  # input w2c, apply w2c
view_points_to_world_points = pose_points_to_world_points  # input w2c, inversely apply w2c


# def grid_sample_blend_weights(grid_coords, bw):
#     # the blend weight is indexed by xyz
#     grid_coords = grid_coords[:, None, None]
#     bw = F.grid_sample(bw,
#                        grid_coords,
#                        padding_mode='border',
#                        align_corners=True)
#     bw = bw[:, :, 0, 0]
#     return bw


# def pts_sample_blend_weights_surf(pts, verts, faces, values) -> torch.Tensor:
#     # surf samp 126988 pts: 127.36531300470233
#     # b, n, D
#     bw, dists = sample_closest_points_on_surface(pts, verts, faces, values)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n


# def pts_sample_blend_weights_vert(pts, verts, values) -> torch.Tensor:
#     # b, n, D
#     bw, dists = sample_closest_points(pts, verts, values)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n


# def pts_sample_blend_weights_vert_blend(pts, verts, values, K=5) -> torch.Tensor:
#     # vert samp K=5 126988 pts: 6.205926998518407
#     # b, n, D
#     bw, dists = sample_blend_K_closest_points(pts, verts, values, K)
#     bw = torch.cat([bw, dists], dim=-1)  # b, n, D+1
#     return bw.permute(0, 2, 1)  # b, D+1, n
# BLENDING


def pts_sample_blend_weights(pts, bw, bounds):
    """sample blend weights for points
    pts: n_batch, n_points, 3
    bw: n_batch, d, h, w, 25
    bounds: n_batch, 2, 3
    """
    pts = pts.clone()

    # interpolate blend weights
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    bounds = max_xyz[:, None] - min_xyz[:, None]
    grid_coords = (pts - min_xyz[:, None]) / bounds
    grid_coords = grid_coords * 2 - 1
    # convert xyz to zyx, since the blend weight is indexed by xyz
    grid_coords = grid_coords[..., [2, 1, 0]]

    # the blend weight is indexed by xyz
    bw = bw.permute(0, 4, 1, 2, 3)
    grid_coords = grid_coords[:, None, None]
    bw = F.grid_sample(bw,
                       grid_coords,
                       padding_mode='border',
                       align_corners=True)
    bw = bw[:, :, 0, 0]

    return bw


def grid_sample_A_blend_weights(nf_grid_coords, bw):
    """
    nf_grid_coords: batch_size x n_samples x 24 x 3
    bw: batch_size x 24 x 64 x 64 x 64
    """
    bws = []
    for i in range(24):
        nf_grid_coords_ = nf_grid_coords[:, :, i]
        nf_grid_coords_ = nf_grid_coords_[:, None, None]
        bw_ = F.grid_sample(bw[:, i:i + 1],
                            nf_grid_coords_,
                            padding_mode='border',
                            align_corners=True)
        bw_ = bw_[:, :, 0, 0]
        bws.append(bw_)
    bw = torch.cat(bws, dim=1)
    return bw


def get_sampling_points(bounds, n_samples):
    sh = bounds.shape
    min_xyz = bounds[:, 0]
    max_xyz = bounds[:, 1]
    x_vals = torch.rand([sh[0], n_samples])
    y_vals = torch.rand([sh[0], n_samples])
    z_vals = torch.rand([sh[0], n_samples])
    vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
    vals = vals.to(bounds.device)
    pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
    return pts


def forward_node_graph(verts: torch.Tensor, graph_rt: torch.Tensor, graph_nodes: torch.Tensor, graph_bones: torch.Tensor, graph_weights: torch.Tensor) -> torch.Tensor:
    n_batch = graph_rt.shape[0]
    verts = verts.expand(n_batch, *verts.shape[1:])
    graph_nodes = graph_nodes.expand(n_batch, *graph_nodes.shape[1:])
    graph_bones = graph_bones.expand(n_batch, *graph_bones.shape[1:])
    graph_weights = graph_weights.expand(n_batch, *graph_weights.shape[1:])

    # graph_bones: B, V, 4
    r, t = graph_rt.split([3, 3], dim=-1)
    R = batch_rodrigues(r.view(-1, 3)).view(n_batch, -1, 3, 3)
    vi = verts[..., None, :].expand(n_batch, -1, graph_bones.shape[-1], -1)  # B, V, 4, 3

    pj = graph_nodes[torch.arange(n_batch)[..., None, None], graph_bones]  # B, V, 4, 3
    tj = t[torch.arange(n_batch)[..., None, None], graph_bones]  # translation B, V, 4, 3
    Rj = R[torch.arange(n_batch)[..., None, None], graph_bones]  # rotation B, V, 4, 3, 3

    wj = graph_weights[..., None].expand(-1, -1, -1, 3)  # B, V, 4, 3
    vj = Rj.matmul((vi - pj)[..., None])[..., 0] + pj + tj  # B, V, 4, 3
    vi = (vj * wj).sum(dim=-2)
    return vi


def forward_deform_lbs(cverts: torch.Tensor, deform: torch.Tensor, weights: torch.Tensor, A: torch.Tensor, R: torch.Tensor = None, T: torch.Tensor = None, big_A=None) -> torch.Tensor:
    n_batch = A.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    if deform is not None:
        tverts = cverts + deform
    else:
        tverts = cverts
    if big_A is not None:
        tverts = pose_points_to_tpose_points(tverts, weights, big_A)
    pverts = tpose_points_to_pose_points(tverts, weights, A)
    if R is not None and T is not None:
        wverts = pose_points_to_world_points(pverts, R, T)
    else:
        wverts = pverts
    return wverts


def inverse_deform_lbs(wverts: torch.Tensor, deform: torch.Tensor, weights: torch.Tensor, A: torch.Tensor, R: torch.Tensor, T: torch.Tensor, big_A=None) -> torch.Tensor:
    n_batch = deform.shape[0]
    weights = weights.expand(n_batch, *weights.shape[1:])
    pverts = world_points_to_pose_points(wverts, R, T)
    tverts = pose_points_to_tpose_points(pverts, weights, A)
    if big_A is not None:
        tverts = tpose_points_to_pose_points(tverts, weights, big_A)
    cverts = tverts - deform
    return cverts


def bilinear_interpolation(input: torch.Tensor, shape: List[int]) -> torch.Tensor:
    # input: B, H, W, C
    # shape: [target_height, target_width]
    return F.interpolate(input.permute(0, 3, 1, 2), shape, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


def rand_sample_sum_to_one(dim, samples, device='cuda', negative_one=False):
    # negative_one: allow sampling to negative one?
    exp_sum = (0.5 * (dim - 1))
    bbweights = torch.rand(samples, dim - 1, device=device)  # 1024, 5
    bbweights_sum = bbweights.sum(dim=-1)
    extra_mask = bbweights_sum > exp_sum
    bbweights[extra_mask] = 1 - bbweights[extra_mask]
    last_row = (bbweights_sum - exp_sum).abs()
    bbweights = torch.cat([bbweights, last_row[..., None]], dim=-1)
    bbweights = bbweights / exp_sum

    if negative_one:
        bbweights = bbweights * 2 - 1 / dim
    return bbweights
    # bbweights = bbweights / (bbweights.sum(dim=-1, keepdim=True) + eps) # MARK: wrong normalization
    # __import__('ipdb').set_trace()


def linear_sample_sum_to_one(dim, samples, device='cuda', multiplier=5.0):
    interval = dim - 1
    samples_per_iter = samples // interval
    samples_last_iter = samples - (interval - 1) * samples_per_iter

    # except last dimension
    weights = torch.zeros(samples, dim, device=device)
    for i in range(interval - 1):
        active = torch.linspace(1, 0, samples_per_iter, device=device)
        active = active - 0.5
        active = active * multiplier
        active = active.sigmoid()
        active = active - 0.5
        active = active / active.max() / 2
        active = active + 0.5
        next = 1 - active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i] = active
        weights[i * samples_per_iter:i * samples_per_iter + samples_per_iter, i + 1] = next

    active = torch.linspace(1, 0, samples_last_iter, device=device)
    next = 1 - active
    weights[(interval - 1) * samples_per_iter:, interval - 1] = active
    weights[(interval - 1) * samples_per_iter:, interval] = next

    return weights


def interpolate_poses(poses, bbweights):
    from pytorch3d import transforms
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, axis_angle_to_matrix
    Rs = axis_angle_to_matrix(poses)
    bbRs: torch.Tensor = torch.einsum('sn,nbij->sbij', bbweights, Rs)
    U, S, Vh = bbRs.svd()
    V = Vh.mH
    # __import__('ipdb').set_trace()
    bbRs = U.matmul(V)
    bbposes = quaternion_to_axis_angle(matrix_to_quaternion(bbRs))
    return bbposes


def interpolate_shapes(shapes, bbweights):
    # bbposes: torch.Tensor = torch.einsum('sn,nvd->svd', bbweights, poses)
    bbshapes: torch.Tensor = torch.einsum('sn,nvd->svd', bbweights, shapes)
    # bbdeformed: torch.Tensor = bbshapes + optim_tpose.verts[None]  # resd to shape
    return bbshapes
