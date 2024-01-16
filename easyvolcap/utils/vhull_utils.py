import torch

from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.chunk_utils import chunkify, multi_gather, multi_scatter
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, torch_inverse_3x3


def multiview_camera_bounds(H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor):
    # msks: N, H, W, 1, the mask to be used
    # c2ws: N, 3, 4, the extrinsic parameters
    # H, W, K: N,; N,; N, 3, 3
    w2c = affine_padding(torch.cat([R, T], dim=-1))  # N, 4, 4
    c2w = affine_inverse(w2c)  # N, 4, 4

    center = c2w[..., :3, -1]  # N, 3
    x_min, y_min, z_min = center[..., 0].min(), center[..., 1].min(), center[..., 2].min()
    x_max, y_max, z_max = center[..., 0].max(), center[..., 1].max(), center[..., 2].max()
    bounds = center.new_tensor([
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
    ])  # 2, 3
    bounds = torch.cat([bounds, bounds[..., :1]], dim=-1)[None] @ w2c.mT  # N, 2, 4 @ N, 4, 4 -> N, 2, 4
    bounds = bounds[..., :-1] / bounds[..., -1:]  # N, 2, 3, camera coordinates
    depth = bounds[..., -1:]  # N, 2, 1

    corners = center.new_tensor([
        [0.0, 0.0],
        [1.0, 1.0],  # in pixel uv coordinates
    ])  # 2, 2
    corners = corners[None] * torch.stack([W, H], dim=-1)[:, None]  # N, 2, 2
    corners = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)  # N, 2, 3
    corners = corners @ torch_inverse_3x3(K).mT  # camera coordinates, N, 2, 3
    corners = corners * depth  # update depth in screen space
    corners = torch.cat([corners, torch.ones_like(corners[..., :1])], dim=-1)  # N, 2, 3
    corners = corners @ c2w.mT  # N, 2, 3 @ N, 4, 4 -> N, 2, 4 -> to world coordinates

    z_min, z_max = corners[..., 2].min(), corners[..., 2].max()
    bounds = center.new_tensor([
        [x_min, y_min, z_min],
        [x_max, y_max, z_max],
    ])  # 2, 3
    return bounds


def create_meshgrid_3d(bounds: torch.Tensor, voxel_size: float = 0.005, indexing='ij'):
    # Define grid points for evaluating the bounding visual hull
    x_min, y_min, z_min = bounds[0]
    x_max, y_max, z_max = bounds[1]
    x = torch.arange(x_min, x_max + voxel_size, voxel_size, device=bounds.device) + voxel_size / 2
    y = torch.arange(y_min, y_max + voxel_size, voxel_size, device=bounds.device) + voxel_size / 2
    z = torch.arange(z_min, z_max + voxel_size, voxel_size, device=bounds.device) + voxel_size / 2
    xyz = torch.meshgrid(x, y, z, indexing=indexing)  # defaults to ij indexing -> W, H, D
    xyz = torch.stack(xyz, dim=-1)  # W, H, D, 3
    return xyz


def reprojection(xyz: torch.Tensor,
                 H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                 msks: torch.Tensor,
                 vhull_thresh: float = 0.75,
                 count_thresh: float = 1,
                 ):
    # Project these points onto the camera
    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)[None]  # homo
    cam = xyz1 @ affine_padding(torch.cat([R, T], dim=-1)).mT  # N, P, 4 @ N, 3, 4 -> N, P, 4, homo camera coordinates
    cam = cam[..., :-1]  # last dim will always be one for valid affine transforms
    pix = cam @ K.mT  # N, P, 3 @ N, 3, 3, homo pixel coords
    pix = pix[..., :-1] / pix[..., -1:]  # N, P, 2, pixel coords

    # Use max size, instead of the one passed in before sampling
    pixel_uv_range = pix / torch.stack([W, H], dim=-1)[..., None, :] * 2 - 1  # N, P, 2 to sample the msk
    should_count_camera = ((pixel_uv_range > -1.0) & (pixel_uv_range < 1.0)).all(dim=-1)  # N, P
    vhull_camera_count = should_count_camera.sum(dim=0)  # P,

    H, W = msks.shape[-3:-1]  # sampling size (the largest of all images)
    # pix = pixel_uv_range
    pix = pix / pix.new_tensor([W, H]) * 2 - 1  # N, P, 2 to sample the msk (dimensionality normalization for sampling)
    valid = F.grid_sample(msks.permute(0, 3, 1, 2), pix[:, None], align_corners=True)[:, 0, 0]  # whether this is background
    valid = (valid > 0.5).float().sum(dim=0)  # N, 1, 1, P -> N, P
    valid = (valid / vhull_camera_count >= vhull_thresh) & (vhull_camera_count >= count_thresh)  # P, ratio of cameras sees this
    return valid


def carve_vhull(
    H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,  # camera parameters
    msks: torch.Tensor,  # the input mask for vhull generation, ..., H, W, 1
    bounds: torch.Tensor,  # the starting bounding box for vhull generation
    padding: float = 0.10,  # the padding to add to the voxels
    voxel_size: float = 0.10,  # the size of the voxel used in vhull generation
    vhull_thresh: float = 0.75,  # the ratio of valid projection in valid cameras (invalid means region out of bound for particular camera)
    count_thresh: int = 1,  # the number of valid camera projection
    remove_outlier: bool = True,
    chunk_size: int = 2097152,
):
    """
    Carve a visual hull from a set of input masks and camera parameters.

    Args:
        H (torch.Tensor): The height of the camera image.
        W (torch.Tensor): The width of the camera image.
        K (torch.Tensor): The intrinsic camera matrix.
        R (torch.Tensor): The rotation matrix for the camera.
        T (torch.Tensor): The translation vector for the camera.
        msks (torch.Tensor): The input mask for vhull generation, with shape ..., H, W, 1.
        bounds (torch.Tensor): The starting bounding box for vhull generation.
        padding (float): The padding to add to the voxels.
        voxel_size (float): The size of the voxel used in vhull generation.
        vhull_thresh (float): The ratio of valid projection in valid cameras (invalid means region out of bound for particular camera).
        count_thresh (int): The number of valid camera projection.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the visual hull and the bounding box for the vhull.
    """
    xyz = create_meshgrid_3d(bounds, voxel_size)
    sh = xyz.shape[:3]
    xyz = xyz.view(-1, 3)  # convert back to P, 3

    @chunkify(chunk_size)
    def chunked_reprojection(xyz, batch: dotdict):
        return reprojection(xyz, **batch)

    valid = chunked_reprojection(xyz, dotdict(H=H,
                                              W=W,
                                              K=K,
                                              R=R,
                                              T=T,
                                              msks=msks,
                                              vhull_thresh=vhull_thresh,
                                              count_thresh=count_thresh,
                                              ))

    # Find valid points on the voxels
    inds = valid.nonzero()
    vhull = multi_gather(xyz, inds)  # P, 3; V -> V, 3
    if remove_outlier:
        from easyvolcap.utils import fcds_utils
        vhull = fcds_utils.remove_outlier(vhull[None], K=5, std_ratio=5.0)[0]

    bounds = torch.stack([vhull.min(dim=0)[0], vhull.max(dim=0)[0]])
    bounds = bounds + bounds.new_tensor([-padding, padding])[:, None]

    return vhull, bounds, valid.view(sh), inds


def hierarchically_carve_vhull(*args,
                               oom_increase_step: float = 0.0005,
                               oom_increase_limit: float = 0.025,
                               **kwargs,
                               ):
    kwargs = dotdict(kwargs)
    voxel_size = kwargs.pop('voxel_size')
    while voxel_size <= oom_increase_limit:
        try:
            return hierarchical_space_carving(*args, **kwargs, voxel_size=voxel_size)
        except RuntimeError as e:
            log(f'OOM: {yellow(e)}, increasing voxel_size to {voxel_size + oom_increase_step}')
            voxel_size = voxel_size + oom_increase_step
            torch.cuda.empty_cache()
            if voxel_size > oom_increase_limit:
                log('Maximum OOM tolerance reached, reraising error')
                raise e


def hierarchical_space_carving(H: torch.Tensor, W: torch.Tensor, K: torch.Tensor, R: torch.Tensor, T: torch.Tensor,
                               msks: torch.Tensor,
                               bounds: torch.Tensor,
                               padding: float = 0.05,
                               voxel_size: float = 0.0333,
                               ctof_factor: float = 3.0,
                               vhull_thresh: float = 0.75,
                               count_thresh: int = 1,  # 10cm voxels for points
                               vhull_thresh_factor: float = 3.0,
                               vhull_count_factor: float = 3.0,
                               coarse_discard_masks: bool = True,
                               intersect_camera_bounds: bool = True,  # this is useful for cicular camera input
                               remove_outlier: bool = True,

                               ):
    """
    Carve a hierarchically refined visual hull from a set of input masks and camera parameters.

    Args:
        H (torch.Tensor): The height of the camera image.
        W (torch.Tensor): The width of the camera image.
        K (torch.Tensor): The intrinsic camera matrix.
        R (torch.Tensor): The rotation matrix for the camera.
        T (torch.Tensor): The translation vector for the camera.
        msks (torch.Tensor): The input mask for vhull generation, with shape ..., H, W, 1.
        bounds (torch.Tensor): The starting bounding box for vhull generation.
        padding (float): The padding to add to the voxels.
        voxel_size (float): The size of the voxel used in vhull generation.
        ctof_factor (float): The factor to convert coarse to fine voxel size.
        vhull_thresh (float): The ratio of valid projection in valid cameras (invalid means region out of bound for particular camera).
        count_thresh (int): The number of valid camera projection.
        vhull_thresh_factor (float): The factor to reduce the vhull threshold for coarse vhull generation.
        vhull_count_factor (float): The factor to reduce the count threshold for coarse vhull generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the hierarchically refined visual hull and the bounding box for the vhull.

    The function carves a hierarchically refined visual hull from a set of input masks and camera parameters. The input masks are used to determine the voxels that are inside the object being reconstructed. The camera parameters are used to project the voxels onto the camera images and determine which voxels are visible from which cameras.

    The function first carves a coarse visual hull by finding the voxels that are visible from a sufficient number of cameras. The function then refines the visual hull by carving in a smaller visual hull size.

    The function returns a tuple containing the hierarchically refined visual hull and the bounding box for the vhull. The bounding box is the smallest axis-aligned box that contains the visual hull.
    """
    # Get the initial voxels
    if intersect_camera_bounds:
        camera_bounds = multiview_camera_bounds(H, W, K, R, T)  # 2, 3
        bounds[0] = torch.maximum(camera_bounds[0], bounds[0])
        bounds[1] = torch.minimum(camera_bounds[1], bounds[1])

    if coarse_discard_masks:
        coarse_masks = torch.ones_like(msks)
    else:
        coarse_masks = msks

    # Carve coarse vhull by finding where all mask pixels blob around
    vhull, bounds, valid, inds = carve_vhull(H, W, K, R, T,
                                             coarse_masks,
                                             bounds,
                                             padding * ctof_factor,
                                             voxel_size * ctof_factor,
                                             min(vhull_thresh / vhull_thresh_factor, 1.0),
                                             count_thresh / vhull_count_factor,
                                             remove_outlier)  # coarse vhull

    # Carve coarse vhull to extract the fine details
    vhull, bounds, valid, inds = carve_vhull(H, W, K, R, T,
                                             msks,
                                             bounds,
                                             padding,
                                             voxel_size,
                                             min(vhull_thresh, 1.0),
                                             count_thresh,
                                             remove_outlier)  # fine vhull
    return vhull, bounds, valid, inds
