# Main
import os
import math
import torch
import random
import numpy as np
import torch.nn.functional as F

from os.path import join
from termcolor import colored
from itertools import accumulate
from collections import defaultdict
from functools import lru_cache, reduce
from smplx.lbs import batch_rodrigues, batch_rigid_transform

from torch import nn
from torch.utils.data import Dataset, get_worker_info
from torch.nn.parallel import DistributedDataParallel as DDP

# Typing
from types import MethodType
from typing import List, Callable, Tuple, Union, Dict

# Utils
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork
    from easyvolcap.models.networks.multilevel_network import MultilevelNetwork

cudart = torch.cuda.cudart()


def indices_to_coords(idx: torch.Tensor, H: int, W: int):
    i = idx // W
    j = idx % W
    return torch.stack([i, j], dim=-1)  # ..., 2


def fair_patch_sample(Hp: int, Wp: int, H: int, W: int):
    """
    Perform sampling of x, y coordinates fairly for patches
    FIXME: Need to compute each pixel's probability of being sampled
    Then sum the inverse of this to the patch center responsible for such samples
    This process could be too slow for efficient and dynamic sampling with difference mask sizes...
    """

    if Hp >= H and Wp >= W:
        # Always 0, 0 if image is smaller than patch
        return 0, 0
    elif Hp >= H or Wp >= W:
        # Multinomial on a line
        if Hp >= H:
            length = W - Wp + 1
            table = torch.arange(0, length)  # sample indices on this line
            diff = Wp  # operate in this room
        else:
            length = H - Hp + 1
            table = torch.arange(0, length)  # sample indices on this line
            diff = Hp  # operate in this room

        # Sample from multinomial
        weight = torch.maximum((diff - table).clip(1), (diff + table - length).clip(1)) ** 0.5
        weight = weight.float().clip(1)
        sample = torch.multinomial(weight, 1).item()

        if Hp > H:
            return sample, 0  # x first
        else:
            return 0, sample
    else:
        # Multinomial on a plane
        table = torch.arange(0, (H - Hp + 1) * (W - Wp + 1)).reshape(H - Hp, W - Wp)  # sample indices on this plane
        ij = indices_to_coords(table, H - Hp, W - Wp)  # ..., 2
        weight_x = torch.maximum((Wp - ij[..., 1]).clip(1), (Wp + ij[..., 1] - H + Hp - 1).clip(1)) ** 0.5
        weight_y = torch.maximum((Hp - ij[..., 0]).clip(1), (Hp + ij[..., 0] - W + Wp - 1).clip(1)) ** 0.5
        weight = weight_x * weight_y
        weight = weight.flatten().float().clip(1)
        sample = torch.multinomial(weight, 1)
        sample = ij[sample].flip(-1)
        return sample[0].item(), sample[1].item()  # x first


def print_shape(batch: dotdict):
    if isinstance(batch, dict):
        for k, v in batch.items():
            print_shape(batch)
    elif isinstance(batch, list):
        for v in batch:
            print_shape(v)
    elif isinstance(v, torch.Tensor):
        print(f'{k}: {v.shape}')
    else:
        print(batch)


def torch_dtype_to_numpy_dtype(torch_dtype):
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.int16: np.int16,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.bool: np.bool_
    }
    return mapping.get(torch_dtype, None)


class DoesNotCareAboutStateDict(nn.Module):
    def __init__(self,):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(f'{prefix}'):
                del state_dict[key]


def register_memory(x: torch.Tensor):
    """
    The implementation of registering memory for fast HtoD copy is ultimately quite tricky due to the fact that
    some of the API for cuda and pytorch are not so robust on Windows (after an Windows 11 Insider update)
    Previously we used torch.cuda.cudart().cudaHostRegister and torch.cuda.cudart().cudaHostUnregister
    Everything is breaking a part, emitting strange errors like FIND CuDNN for convolution & invalid arguments & etc.

    RuntimeError: CUDA error: invalid argument # ???
    RuntimeError: FIND was unable to find an engine to execute this computation # disable benchmarking
    CudaError: part or all of the requested memory range is already mapped (cudaError.???) # make memory contiguous

    However the naive x.pin_memory() is not working either, the registered memory are too large
    Pinning 128MB results in 256MB shared memory usage on Windows, which is not explained anywhere...

    Thus we manually create a new tensor with pin_memory set to True, this should call cudaHostMalloc instead of registering
    This code path is more thoroughly tested by the PyTorch community since the async dataloading involves this
    And as experiments shown, this combines the benefits of the previous two implementations

    And no, it doesn't work.
    """
    # 1:
    # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g564c32d0e6032a9383494b6e63de7bd0
    # x = x.contiguous()
    # torch.cuda.check_error(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0x08))
    # torch.cuda.check_error(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0x0))
    # torch.cuda.synchronize()  # ???
    # torch.cuda.empty_cache()  # ???
    # return x

    # 2:
    # y = torch.empty_like(x, pin_memory=True)  # why extra memory usage here?
    # y = y.copy_(x)
    # return y

    # 3:
    x = x.pin_memory()
    return x

    # TODO: SWITCH THIS BASED ON PLATFORM
    # 4:
    # y = torch.empty_like(x, memory_format=torch.contiguous_format, pin_memory=False)
    # torch.cuda.check_error(cudart.cudaHostRegister(y.data_ptr(), y.numel() * y.element_size(), 0x08))
    # y = y.copy_(x)
    # return y

    # 5:
    # from cuda import cudart
    # CHECK_CUDART_ERROR(cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), cudart.cudaHostRegisterReadOnly))
    # return x


def unregister_memory(x: torch.Tensor):
    # torch.cuda.check_error(cudart.cudaHostUnregister(x.data_ptr()))
    # return x

    # from cuda import cudart
    # CHECK_CUDART_ERROR(cudart.cudaHostUnregister(x.data_ptr()))
    # return x
    y = torch.empty_like(x, pin_memory=False)
    y.copy_(x)
    return x


def FORMAT_CUDART_ERROR(err):
    from cuda import cudart
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def CHECK_CUDART_ERROR(args):
    from cuda import cudart

    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(FORMAT_CUDART_ERROR(err))

    return ret


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


def apply_render_mask(msk: torch.Tensor, *imgs: List[torch.Tensor]):
    # msk: B, H, W, 1
    # imgs: N * [B, H, W, 3]
    # !: This completely ignores the batch dimension
    return imgs


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


def get_xywh_from_mask(msk):
    import torchvision
    X, Y, H, W = 0, 0, *msk.shape[-3:-1]  # EMOTIONAL DAMAGE
    bbox = torchvision.ops.boxes.masks_to_boxes(msk.view(-1, H, W))
    x = bbox[..., 0].min().round().int().item()  # round, smallest of all images
    y = bbox[..., 1].min().round().int().item()  # round, smallest of all images
    w = (bbox[..., 2] - bbox[..., 0]).max().round().int().item()  # round, biggest of all
    h = (bbox[..., 3] - bbox[..., 1]).max().round().int().item()  # round, biggest of all
    return x, y, w, h


def crop_using_mask(msk: torch.Tensor,
                    K: torch.Tensor,
                    *list_of_imgs: List[torch.Tensor]):
    # Deal with empty batch dimension
    bs = msk.shape[:-3]
    msk = msk.view(-1, *msk.shape[-3:])
    K = K.view(-1, *K.shape[-2:])
    list_of_imgs = [im.view(-1, *im.shape[-3:]) for im in list_of_imgs]

    # Assumes channel last format
    # Assumes batch dimension for msk
    # Will crop all images using msk
    # !: EVIL LIST COMPREHENSION
    xs, ys, ws, hs = zip(*[get_xywh_from_mask(m) for m in msk])  # all sizes
    K, *list_of_imgs = zip(*[crop_using_xywh(x, y, w, h, k, *im)
                             for x, y, w, h, k, *im
                             in zip(xs, ys, ws, hs, K, *list_of_imgs)
                             ])  # HACK: This is doable... # outermost: source -> outermost: batch
    K = torch.stack(K)  # stack source dim
    # Resize instead of filling things up?
    # Filling things up might be easier for masked output (bkgd should be black)
    H_max = max(hs)
    W_max = max(ws)
    list_of_imgs = [torch.stack([fill_nhwc_image(im, size=(H_max, W_max)) for im in img]) for img in list_of_imgs]  # HACK: evil list comprehension

    # Restore original dimensionality
    msk = msk.view(*bs, *msk.shape[-3:])
    K = K.view(*bs, *K.shape[-2:])
    list_of_imgs = [im.view(*bs, *im.shape[-3:]) for im in list_of_imgs]
    return K, *list_of_imgs


def crop_using_xywh(x, y, w, h, K, *list_of_imgs):
    K = K.clone()
    K[..., :2, -1] -= torch.as_tensor([x, y], device=K.device)  # crop K
    list_of_imgs = [img[..., y:y + h, x:x + w, :]
                    if isinstance(img, torch.Tensor) else
                    [im[..., y:y + h, x:x + w, :] for im in img]  # HACK: evil list comprehension
                    for img in list_of_imgs]
    return K, *list_of_imgs


class VolumetricVideoModule(nn.Module):
    # This module does not register 'network' as submodule
    def __init__(self, network: nn.Module, collect_timing: bool = False, **kwargs) -> None:
        super().__init__()
        self.unregistered = [network]

        # Prepare timer
        self.timer = Timer(disabled=not collect_timing)

        # Prepare fake forward sample function
        # Hacky forward function definition
        def sample(self, *args, **kwargs):
            if not len(kwargs): batch = args[-1]
            else: batch = kwargs.pop('batch', dotdict())
            self.forward(batch)
            return None, None, None, None
        if not hasattr(self, 'sample'): self.sample = MethodType(sample, self)
        if not hasattr(self, 'render'): self.render = MethodType(sample, self)
        if not hasattr(self, 'compute'): self.compute = MethodType(sample, self)

    @property
    def network(self):
        network: Union["VolumetricVideoNetwork", 'MultilevelNetwork'] = self.unregistered[0]
        return network

    @property
    def collect_timing(self):
        return not self.timer.disabled

    @collect_timing.setter
    def collect_timing(self, value):
        self.timer.disabled = not value


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
    sh = c2w.shape
    pad0 = c2w.new_zeros(sh[:-2] + (1, 3))  # B, 1, 3
    pad1 = c2w.new_ones(sh[:-2] + (1, 1))  # B, 1, 1
    pad = torch.cat([pad0, pad1], dim=-1)  # B, 1, 4
    ext = torch.cat([c2w, pad], dim=-2)  # B, 4, 4
    return ext


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

# MipNeRF360 space contraction


@torch.jit.script
def contract(x: torch.Tensor, r: float = 1.0, p: float = torch.inf):
    l = x.norm(dim=-1, keepdim=True, p=p) + 1e-13
    m = l <= r

    # For smaller than radius points: x = x
    # For larger than radius points: (2 - r/|x|) * r * x / |x|
    x = x * m + ~m * (2 - r / l) * r * x / l
    return x


def reduce_record_stats(record_stats: dotdict):
    reduced_stats = dotdict()
    for k, v in record_stats.items():
        if isinstance(v, torch.Tensor):
            reduced_stats[k] = v.item()  # MARK: will cause sync
        else:
            reduced_stats[k] = v
    return reduced_stats


def schlick_bias(x, s): return (s * x) / ((s - 1) * x + 1)


def schlick_gain(x, s): return torch.where(x < 0.5, schlick_bias(2 * x, s) / 2, (schlick_bias(2 * x - 1, 1 - s) + 1) / 2)


def fill_nchw_image(img: torch.Tensor, size: List[int], value: float = 0.0):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-3:-2]
    zeros = img.new_full((*bs, *cs, *size), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-2], img.shape[-1]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    zeros[..., :h, :w] = img[..., :h, :w]
    return zeros


def fill_nhwc_image(img: torch.Tensor, size: List[int], value: float = 0.0):
    bs = img.shape[:-3]  # -3, -2, -1
    cs = img.shape[-1:]
    zeros = img.new_full((*bs, *size, *cs), value)
    target_h, target_w = size
    source_h, source_w = img.shape[-3], img.shape[-2]
    h = min(target_h, source_h)
    w = min(target_w, source_w)
    zeros[..., :h, :w, :] = img[..., :h, :w, :]
    return zeros


def interpolate_image(img: torch.Tensor, mode='bilinear', align_corners=False, *args, **kwargs):
    # Performs F.interpolate as images (always augment to B, C, H, W)
    sh = img.shape
    img = img.view(-1, *sh[-3:])
    img = F.interpolate(img, *args, mode=mode, align_corners=align_corners if mode != 'nearest' else None, **kwargs)
    img = img.view(sh[:-3] + img.shape[-3:])
    return img


def resize_image(img: torch.Tensor, mode='bilinear', align_corners=False, *args, **kwargs):
    sh = img.shape
    if len(sh) == 4:  # assumption
        img = img.permute(0, 3, 1, 2)
    elif len(sh) == 3:  # assumption
        img = img.permute(2, 0, 1)[None]
    img = interpolate_image(img, mode=mode, align_corners=align_corners, *args, **kwargs)  # uH, uW, 3
    if len(sh) == 4:
        img = img.permute(0, 2, 3, 1)
    elif len(sh) == 3:  # assumption
        img = img[0].permute(1, 2, 0)
    return img


def matchup_channels(t: torch.Tensor, w: torch.Tensor):
    if t.ndim == w.ndim + 1:
        t = t[..., 0]  # remove last dimension
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)  # 65
    return t, w


@torch.jit.script
def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    if x.ndim == xp.ndim - 1:
        x = x[None]

    m = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1] + 1e-8)  # slope
    b = fp[..., :-1] - (m * xp[..., :-1])

    indices = torch.sum(torch.ge(x[..., :, None], xp[..., None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false
    indices = torch.clamp(indices, 0, m.shape[-1] - 1)

    return m.gather(dim=-1, index=indices) * x + b.gather(dim=-1, index=indices)


@torch.jit.script
def integrate_weights(w: torch.Tensor):
    """Compute the cumulative sum of w, assuming all weight vectors sum to 1.
    The output's size on the last dimension is one greater than that of the input,
    because we're computing the integral corresponding to the endpoints of a step
    function, not the integral of the interior/bin values.
    Args:
      w: Tensor, which will be integrated along the last axis. This is assumed to
        sum to 1 along the last axis, and this function will (silently) break if
        that is not the case.
    Returns:
      cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
    """
    cw = torch.cumsum(w[..., :-1], dim=-1).clip(max=1.0)
    shape = cw.shape[:-1] + (1,)
    # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
    cw0 = torch.cat([cw.new_zeros(shape), cw, cw.new_ones(shape)], dim=-1)
    return cw0


@torch.jit.script
def weighted_percentile(t: torch.Tensor, w: torch.Tensor, ps: List[float]):
    """Compute the weighted percentiles of a step function. w's must sum to 1."""
    t, w = matchup_channels(t, w)
    cw = integrate_weights(w)
    # We want to interpolate into the integrated weights according to `ps`.
    # Vmap fn to an arbitrary number of leading dimensions.
    cw_mat = cw.reshape([-1, cw.shape[-1]])
    t_mat = t.reshape([-1, t.shape[-1]])
    wprctile_mat = interpolate(torch.as_tensor(ps).to(t, non_blocking=True),
                               cw_mat,
                               t_mat)
    wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
    return wprctile


def s_vals_to_z_vals(s: torch.Tensor,
                     tn: torch.Tensor,
                     tf: torch.Tensor,
                     g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                     ig: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                     ):
    # transfer ray depth from s space to t space (with inverse of g)
    return ig(s * g(tf) + (1 - s) * g(tn))


def z_vals_to_s_vals(t: torch.Tensor,
                     tn: torch.Tensor,
                     tf: torch.Tensor,
                     g: Callable[[torch.Tensor], torch.Tensor] = lambda x: 1 / x,
                     ):
    # transfer ray depth from t space back to s space (with function g)
    return (g(t) - g(tn)) / (g(tf) - g(tn) + 1e-8)

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

# Hierarchical sampling (section 5.2)


def searchsorted(a: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find indices where v should be inserted into a to maintain order.
    This behaves like jnp.searchsorted (its second output is the same as
    jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
    faster because it wastes memory to save some compute.
    Args:
      a: tensor, the sorted reference points that we are scanning to see where v
        should lie.
      v: tensor, the query points that we are pretending to insert into a. Does
        not need to be sorted. All but the last dimensions should match or expand
        to those of a, the last dimension can differ.
    Returns:
      (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
      range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
      last index of a.
    """
    i = torch.arange(a.shape[-1], device=a.device)  # 128
    v_ge_a = v[..., None, :] >= a[..., :, None]
    idx_lo = torch.max(torch.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)[0]  # 128
    idx_hi = torch.min(torch.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)[0]
    return idx_lo, idx_hi


def invert_cdf(u, t, w):
    """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
    # Compute the PDF and CDF for each weight vector.
    cw = integrate_weights(w)
    # Interpolate into the inverse CDF.
    t_new = interpolate(u, cw, t)
    return t_new


def importance_sampling(t: torch.Tensor,
                        w: torch.Tensor,
                        num_samples: int,
                        perturb=True,
                        single_jitter=False,
                        ):
    """Piecewise-Constant PDF sampling from a step function.

    Args:
        rng: random number generator (or None for `linspace` sampling).
        t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
        w_logits: [..., num_bins], logits corresponding to bin weights
        num_samples: int, the number of samples.
        single_jitter: bool, if True, jitter every sample along each ray by the same
        amount in the inverse CDF. Otherwise, jitter each sample independently.
        deterministic_center: bool, if False, when `rng` is None return samples that
        linspace the entire PDF. If True, skip the front and back of the linspace
        so that the centers of each PDF interval are returned.
        use_gpu_resampling: bool, If True this resamples the rays based on a
        "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
        this resamples the rays based on brute-force searches, which is fast on
        TPUs, but slow on GPUs.

    Returns:
        t_samples: jnp.ndarray(float32), [batch_size, num_samples].
    """
    if t.ndim == w.ndim + 1:
        t = t[..., 0]  # remove last dim

    # preparing for size change
    sh = *t.shape[:-1], num_samples  # B, P, I
    t = t.reshape(-1, t.shape[-1])
    w = w.reshape(-1, w.shape[-1])

    # assuming sampling in s space
    if t.shape[-1] != w.shape[-1] + 1:
        t = torch.cat([t, torch.ones_like(t[..., -1:])], dim=-1)

    # eps = torch.finfo(torch.float32).eps
    eps = 1e-8

    # Draw uniform samples.

    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps if perturb else 0
    d = 1 if single_jitter else num_samples
    u = (
        torch.linspace(0, 1 - u_max, num_samples, device=t.device, dtype=t.dtype) +
        torch.rand(t.shape[:-1] + (d,), device=t.device, dtype=t.dtype) * max_jitter
    )

    u = invert_cdf(u, t, w)

    # preparing for size change
    u = u.reshape(sh)
    return u


def weight_to_pdf(t: torch.Tensor, w: torch.Tensor, eps=torch.finfo(torch.float32).eps**2):
    t, w = matchup_channels(t, w)
    """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
    return w / (t[..., 1:] - t[..., :-1]).clip(eps)


def pdf_to_weight(t: torch.Tensor, p: torch.Tensor):
    t, p = matchup_channels(t, p)
    """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
    return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-torch.inf, torch.inf)):
    t, w = matchup_channels(t, w)
    """Dilate (via max-pooling) a non-negative step function."""
    t0 = t[..., :-1] - dilation
    t1 = t[..., 1:] + dilation
    t_dilate = torch.sort(torch.cat([t, t0, t1], dim=-1), dim=-1)[0]
    t_dilate = t_dilate.clip(*domain)
    w_dilate = torch.max(
        torch.where(
            (t0[..., None, :] <= t_dilate[..., None])
            & (t1[..., None, :] > t_dilate[..., None]),
            w[..., None, :],
            0,
        ),
        dim=-1)[0][..., :-1]
    return t_dilate, w_dilate


def max_dilate_weights(t: torch.Tensor,
                       w: torch.Tensor,
                       dilation: float,
                       domain=(-torch.inf, torch.inf),
                       renormalize=False,
                       eps=torch.finfo(torch.float32).eps**2):
    """Dilate (via max-pooling) a set of weights."""
    p = weight_to_pdf(t, w)
    t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
    w_dilate = pdf_to_weight(t_dilate, p_dilate)
    if renormalize:
        w_dilate /= torch.sum(w_dilate, dim=-1, keepdim=True).clip(eps)
    return t_dilate, w_dilate


def anneal_weights(t: torch.Tensor,
                   w: torch.Tensor,
                   train_frac: float,
                   anneal_slope: float = 10.0,
                   eps=torch.finfo(torch.float32).eps ** 2):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)

    # Optionally anneal the weights as a function of training iteration.
    if anneal_slope > 0:
        # Schlick's bias function, see https://arxiv.org/abs/2010.09714
        def bias(x, s): return (s * x) / ((s - 1) * x + 1)
        anneal = bias(train_frac, anneal_slope)
    else:
        anneal = 1.

    # A slightly more stable way to compute weights**anneal. If the distance
    # between adjacent intervals is zero then its weight is fixed to 0.
    logits_resample = torch.where(
        t[..., 1:] > t[..., :-1],
        anneal * torch.log(w.clip(eps)), -torch.inf)  # MARK: prone to nan

    # If all samples are -inf, softmax will produce a nan (all -torch.inf)
    w = torch.softmax(logits_resample, dim=-1)
    return w


def query(tq, t, y, outside_value=0):
    """Look up the values of the step function (t, y) at locations tq."""
    idx_lo, idx_hi = searchsorted(t, tq)
    yq = torch.where(idx_lo == idx_hi, outside_value,
                     torch.take_along_dim(torch.cat([y, torch.full_like(y[..., :1], outside_value)], dim=-1), idx_lo, dim=-1))  # ?
    return yq


def typed(input_to: torch.dtype = torch.float, output_to: torch.dtype = torch.float):
    from easyvolcap.utils.data_utils import to_x

    def wrapper(func: Callable):
        def inner(*args, **kwargs):
            args = to_x(args, input_to)
            kwargs = to_x(kwargs, input_to)
            ret = func(*args, **kwargs)
            ret = to_x(ret, output_to)
            return ret
        return inner
    return wrapper


def chunkify(chunk_size=1024,
             key='ray_o',
             pos=0,
             dim=-2,
             merge_dims: bool = False,
             ignore_mismatch: bool = False,  # ignore mismatch in batch dims
             print_progress: bool = False,
             move_to_cpu: bool = False,
             batch_key: str = 'batch',
             inds_key: str = 'chunkify_sample',
             ):
    from easyvolcap.utils.data_utils import to_cpu, to_cuda, to_numpy, to_tensor  # keep global imports clean
    # will fail if dim == -1, currently only tested on dim == -2 or dim == 1
    # will select a key element from the argments: either by keyword `key` or position `pos`
    # then, depending on whether user wants to merge other dimensions, will select the dim to chunkify according to `dim`

    def merge_ret(ret, x: torch.Tensor, sh: torch.Size, nn_dim: int):
        # Merge ret list based on reture type (single tensor or dotdict?)
        # Return values of chunified function should all be tensors
        if len(ret) and isinstance(ret[0], torch.Tensor):
            # Stop recursion
            ret = torch.cat(ret, dim=nn_dim)
            if ignore_mismatch:
                ret = ret
            else:
                ret = ret.view(*sh, *ret.shape[nn_dim + 1:]) if x.shape[nn_dim] == ret.shape[nn_dim] else ret
        elif len(ret) and isinstance(ret[0], dict):
            dict_type = type(ret[0])
            # Start recursion
            ret = {k: merge_ret([v[k] for v in ret], x, sh, nn_dim) for k in ret[0].keys()}
            ret = dict_type(ret)
        elif len(ret) and (isinstance(ret[0], list) or isinstance(ret[0], tuple)):
            list_type = type(ret[0])
            # Start recursion
            ret = [merge_ret([v[i] for v in ret], x, sh, nn_dim) for i in range(len(ret[0]))]
            ret = list_type(ret)
        else:
            raise RuntimeError(f'Unsupported return type to batchify: {type(ret[0])}, or got empty return value')
        return ret

    def wrapper(decoder: Callable[[torch.Tensor], torch.Tensor]):
        def decode(*args, **kwargs):
            # Prepare pivot args (find shape information from this arg)
            if key in kwargs:
                x: torch.Tensor = kwargs[key]
            else:
                x: torch.Tensor = args[pos]
                args = [*args]
            sh = x.shape[:dim + 1]  # record original shape up until the chunkified dim
            nn_dim = len(sh) - 1  # make dim a non-negative number (i.e. -2 to 1?)

            # Prepare all tensor arguments by filtering with isinstance
            tensor_args = [v for v in args if isinstance(v, torch.Tensor)]
            tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
            other_args = [v for v in args if not isinstance(v, torch.Tensor)]
            other_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}

            # Merge all dims except first batch dim up until the actual chunkify dimension
            if merge_dims:
                x = x.view(x.shape[0], -1, *x.shape[nn_dim + 1:])
                tensor_args = [v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for v in tensor_args]
                tensor_kwargs = {k: v.view(v.shape[0], -1, *v.shape[nn_dim + 1:]) for k, v in tensor_kwargs.items()}
                nn_dim = 1  # will always be 1 in this situation

            # Running the actual batchified forward pass
            ret = []
            total_size = x.shape[nn_dim]
            # We need to update chunk size so that almost all chunk has a decent amount of queries
            actual_size = math.ceil(total_size / math.ceil(total_size / chunk_size)) if total_size else chunk_size  # this value should be smaller than the actual chunk_size specified
            if print_progress: pbar = tqdm(total=total_size, back=3) # log previous frame
            for i in range(0, total_size, actual_size):
                # nn_dim should be used if there's multiplication involved
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}

                # Other components can use this to perform manual trunking
                if batch_key in other_kwargs: other_kwargs[batch_key].meta[inds_key] = [i, i + actual_size]

                result = decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs)
                result = to_cpu(result, non_blocking=True) if move_to_cpu else result
                ret.append(result)
                if print_progress: pbar.update(min(i + actual_size, total_size) - i)
            if print_progress: pbar.close()  # manual close necessary!

            if not len(ret):
                # Brute-forcely go through the network with empty input
                log(f'zero length tensor detected in chunkify, are the camera parameters correct?', 'red')
                i = 0
                chunk_args = [v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for v in tensor_args]
                chunk_kwargs = {k: v[(slice(None),) * nn_dim + (slice(i, i + actual_size), )] for k, v in tensor_kwargs.items()}
                result = decoder(*chunk_args, *other_args, **chunk_kwargs, **other_kwargs)
                result = to_cpu(result, non_blocking=True) if move_to_cpu else result
                ret.append(result)

            return merge_ret(ret, x, sh, nn_dim)
        return decode
    return wrapper


def key_cache(key: Callable):
    def key_cache_wrapper(func: Callable):
        # will only use argument that match the key positiona or name in the args or kwargs collection as lru_cache's key
        cached_result = None
        cached_hash = None

        def func_wrapper(*args, **kwargs):
            nonlocal cached_result, cached_hash
            key_value = key(*args, **kwargs)
            key_hash = hash(key_value)
            if key_hash != cached_hash:
                cached_result = func(*args, **kwargs)
                cached_hash = key_hash
            return cached_result

        return func_wrapper
    return key_cache_wrapper


def batch_aware_indexing(mask: torch.Tensor, metric: torch.Tensor = None, dim=-1) -> Tuple[torch.Tensor, torch.Tensor, int]:  # MARK: SYNC
    # dim: in terms of the index (mask)
    if mask.dtype != torch.bool: mask = mask.bool()
    if metric is None: metric = mask.int()
    if metric.dtype == torch.bool: metric = metric.int()
    # retain all other dimensions (likely batch dimensions)
    S = mask.sum(dim=dim).max().item()  # the max value of this dim on all other dimension
    valid, inds = metric.topk(S, dim=dim, sorted=False)  # only find the top (mask = True) values (randomly select other values)
    return valid, inds, S


def compute_ground_tris(o: torch.Tensor, d: torch.Tensor):
    n = normalize(torch.rand_like(d))  # B, P, 3,
    a = torch.cross(d, n)
    b = torch.cross(d, a)
    return torch.stack([o, o + a, o + b], dim=-1).mT  # B, P, 3, 3 (considering the right normal)


def torch_dot(x: torch.Tensor, y: torch.Tensor):
    return (x * y).sum(dim=-1)


def angle_to_rotation_2d(theta: torch.Tensor):
    sin = theta.sin()
    cos = theta.cos()
    R = theta.new_zeros(*theta.shape[:-1], 2, 2)
    R[..., 0, :1] = cos
    R[..., 1, :1] = sin
    R[..., 0, 1:] = -sin
    R[..., 1, 1:] = cos

    return R


def multi_indexing(indices: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - indices.ndim
    for _ in range(back_pad): indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)


def multi_gather(values: torch.Tensor, indices: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3

    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(indices, values.shape, dim))


def multi_scatter(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=-2):
    # backward of multi_gather
    return target.scatter(dim, multi_indexing(indices, values.shape, dim), values)


def multi_scatter_(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=-2):
    # inplace version of multi_scatter
    return target.scatter_(dim, multi_indexing(indices, values.shape, dim), values)


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1): f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def linear_indexing(indices: torch.Tensor, shape: torch.Size, dim=0):
    assert indices.ndim == 1
    shape = list(shape)
    dim = dim if dim >= 0 else len(shape) + dim
    front_pad = dim
    back_pad = len(shape) - dim - 1
    for _ in range(front_pad): indices = indices.unsqueeze(0)
    for _ in range(back_pad): indices = indices.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return indices.expand(*expand_shape)


def linear_gather(values: torch.Tensor, indices: torch.Tensor, dim=0):
    # only taking linear indices as input
    return values.gather(dim, linear_indexing(indices, values.shape, dim))


def linear_scatter(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter(dim, linear_indexing(indices, values.shape, dim), values)


def linear_scatter_(target: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter_(dim, linear_indexing(indices, values.shape, dim), values)


def merge01(x: torch.Tensor):
    return x.reshape(-1, *x.shape[2:])


def scatter0(target: torch.Tensor, inds: torch.Tensor, value: torch.Tensor):
    return target.scatter(0, expand_at_the_back(target, inds), value)  # Surface, 3 -> B * S, 3


def gather0(target: torch.Tensor, inds: torch.Tensor):
    return target.gather(0, expand_at_the_back(target, inds))  # B * S, 3 -> Surface, 3


def expand_at_the_back(target: torch.Tensor, inds: torch.Tensor):
    for _ in range(target.ndim - 1):
        inds = inds.unsqueeze(-1)
    inds = inds.expand(-1, *target.shape[1:])
    return inds


def expand0(x: torch.Tensor, B: int):
    return x[None].expand(B, *x.shape)


def expand1(x: torch.Tensor, P: int):
    return x[:, None].expand(-1, P, *x.shape[1:])


def nonzero0(condition: torch.Tensor):
    # MARK: will cause gpu cpu sync
    # return those that are true in the provided tensor
    return condition.nonzero(as_tuple=True)[0]


def get_wsampling_points(ray_o: torch.Tensor, ray_d: torch.Tensor, wpts: torch.Tensor, z_interval=0.01, n_samples=11, perturb=True):
    # calculate the steps for each ray
    z_vals = torch.linspace(-z_interval, z_interval, steps=n_samples, dtype=ray_d.dtype, device=ray_d.device)

    if n_samples == 1:
        z_vals[:] = 0

    if perturb:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, dtype=ray_d.dtype, device=ray_d.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = wpts[:, :, None] + ray_d[:, :, None] * z_vals[:, None]  # B, N, S, 3
    z_vals = (pts[..., 0] - ray_o[..., :1]) / ray_d[..., :1]  # using x dim to calculate depth

    return pts, z_vals

# Grouped convolution for SLRF


def grouped_mlp(I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU(), type='fused'):
    if type == 'fused':
        return FusedGroupedMLP(I, N, W, D, Z, actvn)  # fast, worse, # ? why: problem with grad magnitude
    elif type == 'gconv':
        return GConvGroupedMLP(I, N, W, D, Z, actvn)  # slow, better


class GradientModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradientModule, self).__init__()

    def take_gradient(self, output: torch.Tensor, input: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return take_gradient(output, input, d_out, self.training or create_graph, self.training or retain_graph)

    def take_jacobian(self, output: torch.Tensor, input: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.take_gradient(o, input, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-2)
        return jac


class FusedGroupedMLP(GradientModule):
    # I: input dim
    # N: group count
    # W: network width
    # D: network depth
    # Z: output dim
    # actvn: network activation

    # Fisrt layer: (B, N * I, S) -> (B * N, I, S) -> (B * N, S, I)
    # Weight + bias: (N, I, W) + (N, W) -> pad to (B, N, I, W) -> (B * N, I, W)
    # Result: (B * N, S, W) + (B * N, S, W)
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        super(FusedGroupedMLP, self).__init__()
        self.N = N
        self.I = I
        self.Z = Z
        self.W = W
        self.D = D
        self.actvn = actvn

        self.Is = \
            [I] +\
            [W for _ in range(D - 2)] +\
            [W]\

        self.Zs = \
            [W] +\
            [W for _ in range(D - 2)] +\
            [Z]\

        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(N, I, W))] +
            [nn.Parameter(torch.empty(N, W, W)) for _ in range(D - 2)] +
            [nn.Parameter(torch.empty(N, W, Z))]
        )

        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(N, W))] +
            [nn.Parameter(torch.empty(N, W)) for _ in range(D - 2)] +
            [nn.Parameter(torch.empty(N, Z))]
        )

        for i, w in enumerate(self.weights):  # list stores reference
            ksqrt = np.sqrt(1 / self.Is[i])
            nn.init.uniform_(w, -ksqrt, ksqrt)
        for i, b in enumerate(self.biases):
            ksqrt = np.sqrt(1 / self.Is[i])
            nn.init.uniform_(b, -ksqrt, ksqrt)

    def forward(self, x: torch.Tensor):
        B, N, S, I = x.shape
        x = x.view(B * self.N, S, self.I)

        for i in range(self.D):
            I = self.Is[i]
            Z = self.Zs[i]
            w = self.weights[i]  # N, W, W
            b = self.biases[i]  # N, W

            w = w[None].expand(B, -1, -1, -1).reshape(B * self.N, I, Z)
            b = b[None, :, None].expand(B, -1, -1, -1).reshape(B * self.N, -1, Z)
            x = torch.baddbmm(b, x, w)  # will this just take mean along batch dimension?
            if i < self.D - 1:
                x = self.actvn(x)

        x = x.view(B, self.N, S, self.Z)  # ditching gconv

        return x


class GConvGroupedMLP(GradientModule):
    def __init__(self, I: int, N: int, W: int, D: int, Z: int, actvn: nn.Module = nn.ReLU()):
        # I: input dim
        # N: group count
        # W: network width
        # D: network depth
        # Z: output dim
        # actvn: network activation
        super(GConvGroupedMLP, self).__init__()
        self.mlp = nn.ModuleList(
            [nn.Conv1d(N * I, N * W, 1, groups=N), actvn] +
            [f for f in [nn.Conv1d(N * W, N * W, 1, groups=N), actvn] for _ in range(D - 2)] +
            [nn.Conv1d(N * W, N * Z, 1, groups=N)]
        )
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor):
        # x: B, N, S, C
        B, N, S, I = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B, N * I, S)
        x = self.mlp(x)
        x = x.reshape(B, N, -1, S)
        x = x.permute(0, 1, 3, 2)
        return x


class cVAE(GradientModule):  # group cVAE with grouped convolution
    def __init__(self,
                 group_cnt: int,
                 latent_dim: int,
                 in_dim: int,
                 cond_dim: int,
                 out_dim: int,

                 encode_w: int,
                 encode_d: int,

                 decode_w: int,
                 decode_d: int,
                 ):
        super(cVAE, self).__init__()

        self.N = group_cnt
        self.L = latent_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # input: embedded time concatenated with multiplied pose
        # output: mu and log_var
        I = in_dim + cond_dim
        N, W, D, Z = group_cnt, encode_w, encode_d, latent_dim * 2
        self.encoder = grouped_mlp(I, N, W, D, Z)

        # input: reparameterized latent variable
        # output: high-dim embedding + 3D residual node trans
        I = latent_dim + cond_dim
        N, W, D, Z = group_cnt, decode_w, decode_d, out_dim
        self.decoder = grouped_mlp(I, N, W, D, Z)

    def encode(self, x: torch.Tensor):
        # x: B, N, S, I
        mu, log_var = self.encoder(x).split([self.L, self.L], dim=-1)
        return mu, log_var

    def decode(self, z: torch.Tensor):
        # z: B, N, S, 8
        out = self.decoder(z)
        return out

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        input_ndim = x.ndim
        if input_ndim == 3 and self.N == 1:
            x = x[:, None]
            c = c[:, None]
        elif input_ndim == 2 and self.N == 1:
            x = x[:, None, None]
            c = c[:, None, None]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        # x: B, N, S, C, where C is N * in_dim
        # where in_dim should be embedded time concatenated with multiplied pose
        mu, log_var = self.encode(torch.cat([x, c], dim=-1))  # this second is a lot slower than decode, why?
        z = self.reparameterize(mu, log_var)
        out = self.decode(torch.cat([z, c], dim=-1))
        # out: B, N, S, out_dim(1)
        # mu: B, N, S, 8, log_var: B, N, S, 8, z: B, N, S, 8

        if input_ndim == 3 and self.N == 1:
            out = out[:, 0]
            mu = mu[:, 0]
            log_var = log_var[:, 0]
            z = z[:, 0]
        elif input_ndim == 2 and self.N == 1:
            out = out[:, 0, 0]
            mu = mu[:, 0, 0]
            log_var = log_var[:, 0, 0]
            z = z[:, 0, 0]
        else:
            raise NotImplementedError(f'Unsupported input shape: x.shape: {x.shape}, c.shape: {c.shape} for node count: {self.N}')

        return out, mu, log_var, z

# Resnet Blocks


class ResnetBlock(nn.Module):
    """
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, kernel_size, size_out=None, size_h=None):
        super(ResnetBlock, self).__init__()

        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        padding = kernel_size // 2
        self.conv_0 = nn.Conv2d(size_in, size_h, kernel_size=kernel_size, padding=padding)
        self.conv_1 = nn.Conv2d(size_h, size_out, kernel_size=kernel_size, padding=padding)
        self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(size_in, size_out, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        net = self.conv_0(self.activation(x))
        dx = self.conv_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


def get_function(f: Union[Callable, nn.Module, str]):
    if isinstance(f, str):
        try: return getattr(F, f)  # 'softplus'
        except AttributeError: pass
        try: return getattr(nn, f)()  # 'Identity'
        except AttributeError: pass
        # Using eval is dangerous, will never support that
    else:
        return f


def modulize(f: Callable):
    return Modulized(f)


class Modulized(nn.Module):
    def __init__(self, f: Callable):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor):
        return self.f(x)


def number_of_params(network: nn.Module):
    return sum([p.numel() for p in network.parameters() if p.requires_grad])


def make_params(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=True)


def make_buffer(params: torch.Tensor):
    return nn.Parameter(params, requires_grad=False)


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


def torch_unique_with_indices_and_inverse(x, dim=0):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    indices, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, indices.new_empty(unique.size(dim)).scatter_(dim, indices, perm), inverse


def unmerge_faces(faces: torch.Tensor, *args):
    # stack into pairs of (vertex index, texture index)
    stackable = [faces.reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(arg.reshape(-1))

    # unify them into rows of a numpy array
    stack = torch.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    _, unique, inverse = torch_unique_with_indices_and_inverse(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = torch.zeros(len(order), dtype=torch.long, device=faces.device)
    remap[order] = torch.arange(len(order), device=faces.device)

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def merge_faces(faces, *args, n_verts=None):
    # TODO: batch this
    # remember device the faces are on
    device = faces.device
    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    if n_verts is None:  # sometimes things get padded
        n_verts = faces.max() + 1
    # add a vertex mask which is just ordered
    result.append(torch.arange(n_verts, device=device))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = torch.zeros((3, n_verts), dtype=torch.long, device=device)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.permute(*torch.arange(faces.ndim - 1, -1, -1)), arg.permute(*torch.arange(arg.ndim - 1, -1, -1))):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(torch.median(masks, dim=0)[0].to(torch.long))

    return result


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


def pad_layer_output(layer_output: dotdict):
    # Fetch the layer output
    HrWr, xywh = layer_output.HrWr, layer_output.xywh
    # Complete the foreground layer object bounding box
    output = dotdict()

    for key, value in layer_output.items():
        # Complete a specific output of all layers
        if key in ['HrWr', 'xywh', 'layer_render']: continue

        # Nasty shape, especially in layered representation
        H, W = HrWr[-1]  # !: H, W should be the same for all foreground layers
        B, _, N, C = value[-1].shape if len(value[-1].shape) == 4 else value[-1][..., None].shape  # N may differ for different foreground layers
        x, y, w, h = xywh[-1]  # x, y, w, h of the foreground object bounding box, may differ

        # Actual complete logic
        value_c = torch.zeros(B, H, W, N, C).to(value[-1].device)  # (B, H, W, N, C)
        value_c[:, y:y + h, x:x + w] = value[-1].reshape(B, h, w, N, C)  # (B, H, W, N, C)
        output[key] = value_c.reshape(B, -1, N, C)

    return output


def pad_layers_output(layer_output: dotdict):
    # Fetch the layer output
    HrWr, xywh = layer_output.HrWr, layer_output.xywh
    # Complete the foreground layer object bounding box
    output = dotdict()

    # Concatenate all layers output
    for key, value in layer_output.items():
        # Complete a specific output of all layers
        if key in ['HrWr', 'xywh', 'layer_render']: continue
        else: output[key] = []

        for i in range(len(value)):
            # Nasty shape, especially in layered representation
            H, W = HrWr[i]  # !: H, W should be the same for all foreground layers
            B, _, N, C = value[i].shape if len(value[i].shape) == 4 else value[i][..., None].shape  # N may differ for different foreground layers
            x, y, w, h = xywh[i]  # x, y, w, h of the foreground object bounding box, may differ

            # Actual complete logic
            value_c = torch.zeros(B, H, W, N, C).to(value[i].device)  # (B, H, W, N, C)
            value_c[:, y:y + h, x:x + w] = value[i].reshape(B, h, w, N, C)  # (B, H, W, N, C)
            output[key].append(value_c.reshape(B, -1, N, C))

        # Concatenate all layers output together
        output[key] = torch.cat(output[key], dim=-2)  # (B, H * W, N * num_fg_layer, C)

    # Sort the z_vals and get the indices
    output.z_vals, indices = torch.sort(output.z_vals, dim=-2)  # (B, H * W, N * num_fg_layer, 1)
    # Rearrange the other outputs according to the indices
    for key, value in output.items():
        if key in ['z_vals', 'HrWr', 'xywh', 'layer_render']: continue
        else: output[key] = torch.gather(value, dim=-2, index=indices.expand(-1, -1, -1, value.shape[-1]))

    return output


def volume_rendering_layer(output: dotdict, rgb: torch.Tensor, occ: torch.Tensor, bg_brightness=0.0):
    # See volume_rendering for more intuition
    # Concatenate the background rgb and occ to the last
    rgbs = torch.cat([output.rgb, rgb], dim=-2)  # (B, H * W, N * num_fg_layer + N, 3)
    occs = torch.cat([output.occ, occ], dim=-2)  # (B, H * W, N * num_fg_layer + N, 3)

    weights, rgb_map, acc_map = volume_rendering(rgbs, occs, bg_brightness)  # (B, H * W, N * num_fg_layer + N), (B, H * W, 3), (B, H * W, 1)
    return weights, rgb_map, acc_map


def clamped_logit(x, high=1.0):
    x_high = torch.sigmoid(torch.tensor(high, dtyp=x.dtype, device=x.device))
    # scale = 1 - 2 * (1 - x_high)
    scale = 2 * x_high - 1
    x /= 2
    x -= 0.5
    x *= scale
    x += 0.5
    return torch.logit(x)


def shift_occ_to_msk(occ: torch.Tensor, min: float = 0.0, max: float = 1.0):
    # occ: (b, n, s), but indicates occupancy, [min, max], min outside, max inside
    msk = occ.max(dim=-1)[0]  # min_val, min_ind, find max along all samples
    msk -= min  # move
    msk /= max - min  # norm
    return msk


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


def expand_result_to_query_shape(func):
    def wrapper(*arg, **kwargs):
        query = arg[1]  # 0 is self, 1 is actually the first input
        val, ret = func(*arg, **kwargs)
        full = torch.zeros(*query.shape[:-1], val.shape[-1], device=val.device, dtype=val.dtype)
        full[ret.inds] = val
        return full
    return wrapper


def expand_result_to_query_shape_as_raw(func):
    def wrapper(*arg, **kwargs):
        query = arg[1]  # 0 is self, 1 is actually the first input
        val, ret = func(*arg, **kwargs)
        full = torch.zeros(*query.shape[:-1], val.shape[-1], device=val.device, dtype=val.dtype)
        full[ret.inds] = val
        ret.raw = full
        return ret
    return wrapper


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
    from smplx.lbs import batch_rigid_transform, batch_rodrigues
    # pose: B, N, 3
    # joints: B, N, 3
    # parents: B, N
    # B, N, _ = poses.shape
    R = batch_rodrigues(poses.view(-1, 3))  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints, parents.view(-1))  # MARK: doc of this is wrong about parent
    return J, A


def get_rigid_transform_nobatch(poses: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor):
    from smplx.lbs import batch_rigid_transform, batch_rodrigues
    # pose: N, 3
    # joints: N, 3
    # parents: N
    R = batch_rodrigues(poses)  # N, 3, 3
    J, A = batch_rigid_transform(R[None], joints[None], parents)  # MARK: doc of this is wrong about parent
    J, A = J[0], A[0]  # remove batch dimension
    return J, A


def fast_sample_rays(ray_o, ray_d, near, far, img, msk, mask_at_box, n_rays, split='train', body_ratio=0.5, face_ratio=0.0):
    msk = msk * mask_at_box
    if "train" in split:
        n_body = int(n_rays * body_ratio)
        n_face = int(n_rays * face_ratio)
        n_rays = n_rays - n_body - n_face
        coord_body = torch.nonzero(msk == 1)
        coord_face = torch.nonzero(msk == 13)
        coord_rand = torch.nonzero(mask_at_box == 1)

        coord_rand = coord_rand[torch.randint(len(coord_rand), (n_rays, ))]
        if len(coord_body): coord_body = coord_body[torch.randint(len(coord_body), (n_body, ))]
        if len(coord_face): coord_face = coord_face[torch.randint(len(coord_face), (n_face, ))]

        coord = coord_rand
        if len(coord_body): coord = torch.cat([coord_body, coord], dim=0)
        if len(coord_face): coord = torch.cat([coord_face, coord], dim=0)

        mask_at_box = mask_at_box[coord[:, 0], coord[:, 1]]  # always True when training
    else:
        coord = torch.nonzero(mask_at_box == 1)
        # will not modify mask at box
    ray_o = ray_o[coord[:, 0], coord[:, 1]]
    ray_d = ray_d[coord[:, 0], coord[:, 1]]
    near = near[coord[:, 0], coord[:, 1]]
    far = far[coord[:, 0], coord[:, 1]]
    rgb = img[coord[:, 0], coord[:, 1]]
    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def take_jacobian(func: Callable, input: torch.Tensor, create_graph=False, vectorize=True, strategy='reverse-mode'):
    return torch.autograd.functional.jacobian(func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class MLP(GradientModule):
    def __init__(self, input_ch=32, W=256, D=8, out_ch=257, skips=[4], actvn=nn.ReLU(), out_actvn=nn.Identity(),
                 init_weight=nn.Identity(), init_bias=nn.Identity(), init_out_weight=nn.Identity(), init_out_bias=nn.Identity(), dtype=torch.float):
        super(MLP, self).__init__()
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.skips = skips
        self.linears = []
        for i in range(D + 1):
            I, O = W, W
            if i == 0:
                I = input_ch
            if i in skips:
                I = input_ch + W
            if i == D:
                O = out_ch
            self.linears.append(nn.Linear(I, O, dtype=dtype))
        self.linears = nn.ModuleList(self.linears)
        self.actvn = get_function(actvn) if isinstance(actvn, str) else actvn
        self.out_actvn = get_function(out_actvn) if isinstance(out_actvn, str) else out_actvn

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1: init_out_weight(l.weight.data)
            else: init_weight(l.weight.data)

        for i, l in enumerate(self.linears):
            if i == len(self.linears) - 1: init_out_bias(l.bias.data)
            else: init_bias(l.bias.data)

    def forward_with_previous(self, input: torch.Tensor):
        x = input
        for i, l in enumerate(self.linears):
            p = x  # store output of previous layer
            if i in self.skips:
                x = torch.cat([x, input], dim=-1)
            if i == len(self.linears) - 1:
                a = self.out_actvn
            else:
                a = self.actvn
            x = a(l(x))  # actual forward
        return x, p

    def forward(self, input: torch.Tensor):
        return self.forward_with_previous(input)[0]


class SphereSignedDistanceField(GradientModule):
    def __init__(self, d_in=63, d_hidden=256, n_layers=8, d_out=257, skips=[4], embedder=None):
        super(SphereSignedDistanceField, self).__init__()
        if embedder is not None:
            self.embedder = embedder

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        skips = [4]
        bias = 0.5
        scale = 1.0
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skips
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if hasattr(self, 'embedder'):
            inputs = self.embedder(inputs)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], dim=-1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[..., :1] / self.scale, x[..., 1:]], dim=-1)


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


def setup_deterministic(fix_random=True, # all deterministic, same seed, no benchmarking
                        allow_tf32=False, # use tf32 support if possible
                        deterministic=True, # use deterministic algorithm for CNN
                        benchmark=False, 
                        seed=0, # only used when fix random is set to true
                        ):
    # https://huggingface.co/docs/diffusers/v0.9.0/en/optimization/fp16
    # https://huggingface.co/docs/transformers/v4.18.0/en/performance#tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32 # by default, tf32 support of CNNs is enabled
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    if fix_random:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)


def get_state_dict(state_dict: dotdict, prefix: str = ''):
    if len(prefix) and not prefix.endswith('.'): prefix = prefix + '.'
    d = dotdict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            d[k[len(prefix):]] = v
    return d


def load_pretrained(model_dir: str, resume: bool = True, epoch: int = -1, ext: str = '.npz', remove_if_not_resuming: bool = False, warn_if_not_exist: bool = False):
    if not resume:  # remove nothing here
        if remove_if_not_resuming:
            if os.path.isdir(model_dir) and len(os.listdir(model_dir)):  # only inform the use if there are files
                # log(red(f"Removing trained weights: {blue(model_dir)}"))
                try: run(f'rm -r {model_dir}')
                except: pass
        return None, None

    if not os.path.exists(model_dir):
        if warn_if_not_exist:
            log(red(f'Pretrained network: {blue(model_dir)} does not exist'))
        return None, None
    if os.path.isdir(model_dir):
        pts = [
            int(pt.split('.')[0]) for pt in os.listdir(model_dir) if pt != f'latest{ext}' and pt.endswith(ext) and pt.split('.')[0].isnumeric()
        ]
        if len(pts) == 0 and f'latest{ext}' not in os.listdir(model_dir):
            return None, None
        if epoch == -1:
            if f'latest{ext}' in os.listdir(model_dir):
                pt = 'latest'
            else:
                pt = max(pts)
        else:
            pt = epoch
        model_path = join(model_dir, f'{pt}{ext}')
    else:
        model_path = model_dir

    if ext == '.pt' or ext == '.pth':
        pretrained = dotdict(torch.load(model_path, 'cpu'))
    else:
        from easyvolcap.utils.data_utils import to_tensor
        pretrained = dotdict(model=to_tensor(dict(**np.load(model_path))), epoch=-1)  # the npz files do not contain training parameters

    return pretrained, model_path


def load_model(
    model: nn.Module,
    optimizer: Union[nn.Module, None] = None,
    scheduler: Union[nn.Module, None] = None,
    moderator: Union[nn.Module, None] = None,
    model_dir: str = '',
    resume: bool = True,
    epoch: int = -1,
    strict: bool = True,  # report errors when loading "model" instead of network
    skips: List[str] = [],
    only: List[str] = [],
    allow_mismatch: List[str] = [],
):

    pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pt',
                                             remove_if_not_resuming=True,
                                             warn_if_not_exist=False)
    if pretrained is None: return 0

    pretrained_model = pretrained['model']
    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(model_parent, last_name, nn.Parameter(pretrained_parent[last_name], requires_grad=getattr(model_parent, last_name).requires_grad))  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(pretrained_model, strict=strict)
    if optimizer is not None and 'optimizer' in pretrained.keys(): optimizer.load_state_dict(pretrained['optimizer'])
    if scheduler is not None and 'scheduler' in pretrained.keys(): scheduler.load_state_dict(pretrained['scheduler'])
    if moderator is not None and 'moderator' in pretrained.keys(): moderator.load_state_dict(pretrained['moderator'])
    log(f'Loaded model {blue(model_path)} at epoch {blue(pretrained["epoch"])}')
    return pretrained['epoch'] + 1


def load_network(
    model: nn.Module,
    model_dir: str = '',
    resume: bool = True,  # when resume is False, will try as a fresh restart
    epoch: int = -1,
    strict: bool = True,  # report errors if something is wrong
    skips: List[str] = [],
    only: List[str] = [],
    prefix: str = '',  # will match and remove these prefix
    allow_mismatch: List[str] = [],
):
    pretrained, model_path = load_pretrained(model_dir, resume, epoch,
                                             remove_if_not_resuming=False,
                                             warn_if_not_exist=False)
    if pretrained is None:
        pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pth',
                                                 remove_if_not_resuming=False,
                                                 warn_if_not_exist=False)
    if pretrained is None:
        pretrained, model_path = load_pretrained(model_dir, resume, epoch, '.pt',
                                                 remove_if_not_resuming=False,
                                                 warn_if_not_exist=resume)
    if pretrained is None:
        return 0

    # log(f'Loading network: {blue(model_path)}')
    # ordered dict cannot be mutated while iterating
    # vanilla dict cannot change size while iterating
    pretrained_model = pretrained['model']

    if skips:
        keys = list(pretrained_model.keys())
        for k in keys:
            if root_of_any(k, skips):
                del pretrained_model[k]

    if only:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if not root_of_any(k, only):
                del pretrained_model[k]

    if prefix:
        keys = list(pretrained_model.keys())  # since the dict has been mutated, some keys might not exist
        for k in keys:
            if k.startswith(prefix):
                pretrained_model[k[len(prefix):]] = pretrained_model[k]
            del pretrained_model[k]

    for key in allow_mismatch:
        if key in model.state_dict() and key in pretrained_model and not strict:
            model_parent = model
            pretrained_parent = pretrained_model
            chain = key.split('.')
            for k in chain[:-1]:  # except last one
                model_parent = getattr(model_parent, k)
                pretrained_parent = pretrained_parent[k]
            last_name = chain[-1]
            setattr(model_parent, last_name, nn.Parameter(pretrained_parent[last_name], requires_grad=getattr(model_parent, last_name).requires_grad))  # just replace without copying

    (model if not isinstance(model, DDP) else model.module).load_state_dict(pretrained_model, strict=strict)
    log(f'Loaded network {blue(model_path)} at epoch {blue(pretrained["epoch"])}')
    return pretrained["epoch"] + 1


def save_npz(model: nn.Module,
             model_dir: str = '',
             epoch: int = -1,
             latest: int = True,
             ):
    from easyvolcap.utils.data_utils import to_numpy
    npz_path = join(model_dir, 'latest.npz' if latest else f'{epoch}.npz')
    state_dict = model.state_dict() if not isinstance(model, DDP) else model.module.state_dict()
    param_dict = to_numpy(state_dict)  # a shallow dict
    os.makedirs(dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **param_dict)
    log(yellow(f'Saved model {blue(npz_path)} at epoch {blue(epoch)}'))


def save_model(model: nn.Module,
               optimizer: Union[nn.Module, None] = None,
               scheduler: Union[nn.Module, None] = None,
               moderator: Union[nn.Module, None] = None,
               model_dir: str = '',
               epoch: int = -1,
               latest: int = False,
               save_lim: int = 5,
               ):

    model = {
        # Special handling for ddp modules (incorrect naming)
        'model': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
        'epoch': epoch
    }

    if optimizer is not None:
        model['optimizer'] = optimizer.state_dict()

    if scheduler is not None:
        model['scheduler'] = scheduler.state_dict()

    if moderator is not None:
        model['moderator'] = moderator.state_dict()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_path = join(model_dir, 'latest.pt' if latest else f'{epoch}.pt')
    torch.save(model, model_path)
    log(yellow(f'Saved model {blue(model_path)} at epoch {blue(epoch)}'))

    pts = [
        int(pt.split('.')[0]) for pt in os.listdir(model_dir)
        if pt != 'latest.pt' and pt.endswith('.pt')
    ]
    if len(pts) <= save_lim:
        return
    else:
        removing = join(model_dir, f"{min(pts)}.pt")
        # log(red(f"Removing trained weights: {blue(removing)}"))
        os.remove(removing)


def root_of_any(k, l):
    for s in l:
        a = accumulate(k.split('.'), lambda x, y: x + '.' + y)
        for r in a:
            if s == r:
                return True
    return False


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


def reset_optimizer_state(optimizer):
    optimizer.__setstate__({'state': defaultdict(dict)})


def update_optimizer_state(optimizer, optimizer_state):
    for k, v in optimizer_state.items():
        if v.new_params == None:
            continue
        val = optimizer.state[k].copy()
        exp_avg = torch.zeros_like(v.new_params)
        exp_avg[v.new_keep] = val['exp_avg'][v.old_keep]
        val['exp_avg'] = exp_avg
        exp_avg_sq = torch.zeros_like(v.new_params)
        exp_avg_sq[v.new_keep] = val['exp_avg_sq'][v.old_keep]
        val['exp_avg_sq'] = exp_avg_sq
        del optimizer.state[k]
        optimizer.state[v.new_params] = val


def logits_to_prob(logits):
    ''' Returns probabilities for logits
    Args:
        logits (tensor): logits
    '''
    odds = torch.exp(logits)
    probs = odds / (1 + odds)
    return probs


def prob_to_logits(probs, eps=1e-4):
    ''' Returns logits for probabilities.
    Args:
        probs (tensor): probability tensor
        eps (float): epsilon value for numerical stability
    '''
    probs = torch.clip(probs, a_min=eps, a_max=1 - eps)
    logits = torch.log(probs / (1 - probs))
    return logits


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


def get_max_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 20


@torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


@torch.jit.script
def normalize_sum(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.sum(dim=-1, keepdim=True) + eps)


def sigma_to_alpha(raw, dists=0.005, act_fn=F.softplus): return 1. - torch.exp(-act_fn(raw) * dists)


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


def weighted_sample_rays(rgb: torch.Tensor,  # rgb image
                         msk: torch.Tensor,  # mask value (not same as weight)
                         wet: torch.Tensor,  # weight of every pixel (high weights -> sample more)
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
    rgb = rgb[i, j]  # this is unavoidable (need to read from data)
    msk = msk[i, j]  # just whatever for now
    wet = wet[i, j]  # just whatever for now

    # Maybe refactor these? Built rays directly
    # with timeit(get_rays_from_ij.__name__):
    ray_o, ray_d = get_rays_from_ij(i, j, K, R, T, use_z_depth=use_z_depth, correct_pix=correct_pix)  # MARK: 0.xms
    return rgb, msk, wet, ray_o, ray_d, coords  # gt, input, input, others


def weighted_sample_coords(wet: torch.Tensor, n_rays: int = -1) -> torch.Tensor:
    # Here, msk is a sampling weight controller, and only those with >0 values will be sampled
    def indices_to_coords(idx: torch.Tensor, H, W):
        i = idx // W
        j = idx % W
        return torch.stack([i, j], dim=-1)  # ..., 2

    def coords_to_indices(ij: torch.Tensor, H, W):
        i, j = ij.unbind(-1)
        return i * W + j  # ... (no last dim)

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


def get_bound_proposal(bounds: torch.Tensor, ray_o: torch.Tensor, ray_d: torch.Tensor, n_steps: int):
    near, far, mask_at_box = get_near_far_aabb(bounds, ray_o, ray_d)  # near, far already masked
    near, far = near[..., 0], far[..., 0]
    mask_at_box = near < far  # (b, n)
    # box mask
    # filter near far based on box
    near = near[mask_at_box]  # (b, m)
    far = far[mask_at_box]  # (b, m)
    dists = sample_depth_near_far(near, far, n_steps, perturb=False)  # pts already masked
    return near, far, dists, mask_at_box


def compute_norm_diff(surf_pts: torch.Tensor, batch: dotdict[str, torch.Tensor], grad_decoder, diff_range: float, epsilon: float = 1e-6):
    n_batch, n_pts, D = surf_pts.shape
    surf_pts_neighbor = surf_pts + (torch.rand_like(surf_pts) - 0.5) * diff_range
    grad_pts = torch.cat([surf_pts, surf_pts_neighbor], dim=1)  # cat in n_masked dim
    grad: torch.Tensor = grad_decoder(grad_pts, batch)  # (n_batch, n_masked, 3)
    norm = grad / (grad.norm(dim=-1, keepdim=True) + epsilon)  # get normal direction
    norm_diff = (norm[:, n_pts:, :] - norm[:, :n_pts, :])  # neighbor - surface points

    return norm_diff, grad


def compute_val_pair_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    n_batch, n_pts, D = pts.shape
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
    raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    return raw


def compute_diff_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float, norm_value: List[bool] = [True], dims: List[int] = [3]):
    # sample around input point and compute values, then take their difference
    # values are normalized based on settings (norm_value bool list)
    # pts and its random neighbor are concatenated in second dimension
    n_batch, n_pts, D = pts.shape
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
    raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    diff = []
    for i, d in enumerate(dims):
        start = sum(dims[:i])
        stop = start + d
        part_value = raw[:, :, start:stop]
        if norm_value[i]:
            normed_value = normalize(part_value)
        part_diff = (normed_value[:, n_pts:, :] - normed_value[:, :n_pts, :])  # neighbor - surface points -> B, N, D
        diff.append(part_diff)
    diff = torch.cat(diff, dim=-1)

    return diff, raw
