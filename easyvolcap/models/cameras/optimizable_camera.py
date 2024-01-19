# This file contains optimizable camera parameters
# Implemented in SO3xR3, exponential map of rotation and translation from screw rt motion

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from copy import copy
from os.path import join
from easyvolcap.engine import cfg
from easyvolcap.engine import CAMERAS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.lie_utils import exp_map_SO3xR3
from easyvolcap.utils.net_utils import make_params, freeze_module, load_network
from easyvolcap.utils.math_utils import affine_padding, affine_inverse, vector_padding, point_padding


@CAMERAS.register_module()
class OptimizableCamera(nn.Module):
    # TODO: Implement intrinsics optimization
    # MARK: EVIL GLOBAL CONFIG
    bounds = cfg.dataloader_cfg.dataset_cfg.bounds if 'bounds' in cfg.dataloader_cfg.dataset_cfg else [[-1, -1, -1], [1, 1, 1]]  # only used for initialization
    bounds = torch.as_tensor(bounds, dtype=torch.float)

    center = bounds.sum(-2) / 2
    radius = (bounds[1] - bounds[0]).max() / 2
    square_bounds = torch.stack([center - radius, center + radius])

    data_root = cfg.dataloader_cfg.dataset_cfg.data_root if 'data_root' in cfg.dataloader_cfg.dataset_cfg else ''
    vhulls_dir = cfg.dataloader_cfg.dataset_cfg.vhulls_dir if 'vhulls_dir' in cfg.dataloader_cfg.dataset_cfg else 'vhulls'
    images_dir = cfg.dataloader_cfg.dataset_cfg.images_dir if 'images_dir' in cfg.dataloader_cfg.dataset_cfg else 'images'

    view_sample = cfg.dataloader_cfg.dataset_cfg.view_sample if 'view_sample' in cfg.dataloader_cfg.dataset_cfg else [0, None, 1]
    frame_sample = cfg.dataloader_cfg.dataset_cfg.frame_sample if 'frame_sample' in cfg.dataloader_cfg.dataset_cfg else [0, None, 1]

    cams = os.listdir(join(data_root, images_dir)) if exists(join(data_root, images_dir)) else []
    view_sample, frame_sample = copy(view_sample), copy(frame_sample)
    if len(frame_sample) == 3:
        frame_sample[1] = frame_sample[1] or (len(os.listdir(join(data_root, images_dir, cams[0]))) if len(cams) else 1)  # will error out if using this module
        n_frames = (frame_sample[1] - frame_sample[0]) // frame_sample[2]
    else:
        n_frames = len(frame_sample)
    if len(view_sample) == 3:
        view_sample[1] = view_sample[1] or len(cams)  # will error out if using this module
        n_views = (view_sample[1] - view_sample[0]) // view_sample[2]  # FIXME: DIFFERENT MEANING
    else:
        n_views = len(view_sample)
    intri_file = cfg.dataloader_cfg.dataset_cfg.intri_file if 'intri_file' in cfg.dataloader_cfg.dataset_cfg else 'intri.yml'
    extri_file = cfg.dataloader_cfg.dataset_cfg.extri_file if 'extri_file' in cfg.dataloader_cfg.dataset_cfg else 'extri.yml'

    # TODO: Remove the closest using t setting
    closest_using_t = cfg.dataloader_cfg.dataset_cfg.closest_using_t if 'closest_using_t' in cfg.dataloader_cfg.dataset_cfg else False
    moves_through_time = not exists(join(data_root, intri_file)) or not exists(join(data_root, extri_file))

    def __init__(self,
                 n_views: int = n_views,
                 n_frames: int = n_frames,
                 moves_through_time: bool = moves_through_time,
                 pretrained_camera: str = '',
                 freeze_camera: bool = False,
                 dtype: str = 'float',
                 **kwargs,
                 ):
        super().__init__()
        self.n_views = n_views
        self.n_frames = n_frames if moves_through_time else 1
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.pose_resd = make_params(torch.zeros(self.n_frames, self.n_views, 6, dtype=self.dtype))  # F, V, 6

        if os.path.exists(pretrained_camera):
            load_network(self, pretrained_camera, prefix='camera.')  # will load some of the parameters from this model

        if freeze_camera:
            freeze_module(self)

    def forward_srcs(self, batch: dotdict):
        s_inds = batch.src_inds  # B, S, selected source views
        t_inds = batch.t_inds
        if OptimizableCamera.closest_using_t:
            f_inds = s_inds.clip(0, self.n_frames - 1)
            v_inds = t_inds.clip(0, self.n_views - 1)
        else:
            f_inds = t_inds.clip(0, self.n_frames - 1)
            v_inds = s_inds.clip(0, self.n_views - 1)

        pose_resd = self.pose_resd[f_inds, v_inds].to(batch.src_exts)  # B, S, 3, 4
        pose_resd = exp_map_SO3xR3(pose_resd.detach())  # do not optimize through sampling, unstable
        w2c_resd = affine_padding(pose_resd)  # B, S, 3, 4
        w2c_opt = w2c_resd @ batch.src_exts
        batch.src_exts = w2c_opt
        # UNUSED: This is not used for now
        # batch.meta.src_exts = batch.src_exts.to('cpu', non_blocking=True)
        return batch

    def forward_pose(self, batch: dotdict):
        if 'w2c_ori' in batch and 'w2c_opt' in batch and 'w2c_resd' in batch:
            return batch.w2c_ori, batch.w2c_opt, batch.w2c_resd

        view_index = batch.view_index  # B, # avoid synchronization
        latent_index = batch.latent_index  # B,

        view_index = view_index.clip(0, self.n_views - 1)
        latent_index = latent_index.clip(0, self.n_frames - 1)  # TODO: FIX VIEW AND CAMERA AND FRAME AND LATEN INDICES

        pose_resd = self.pose_resd[latent_index, view_index].to(batch.R)  # fancy indexing? -> B, 6
        pose_resd = exp_map_SO3xR3(pose_resd)  # B, 3, 4

        # Use left multiplication
        w2c_resd = affine_padding(pose_resd)  # B, 3, 4
        w2c_ori = affine_padding(torch.cat([batch.R, batch.T], dim=-1))  # B, 3, 4
        w2c_opt = w2c_resd @ w2c_ori

        batch.w2c_ori = w2c_ori
        batch.w2c_opt = w2c_opt
        batch.w2c_resd = w2c_resd
        return w2c_ori, w2c_opt, w2c_resd

    def forward_cams(self, batch: dotdict):
        w2c_ori, w2c_opt, w2c_resd = self.forward_pose(batch)
        batch.orig_R = batch.R
        batch.orig_T = batch.T
        batch.R = w2c_opt[..., :3, :3]
        batch.T = w2c_opt[..., :3, 3:]

        batch.meta.meta_stream = torch.cuda.Stream()
        batch.meta.meta_stream.wait_stream(torch.cuda.current_stream())  # stream synchronization matters
        with torch.cuda.stream(batch.meta.meta_stream):
            batch.meta.R = batch.R.to('cpu', non_blocking=True)
            batch.meta.T = batch.T.to('cpu', non_blocking=True)
        return batch

    def forward_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, batch):
        w2c_ori, w2c_opt, w2c_resd = self.forward_pose(batch)
        inv_w2c_opt = affine_inverse(w2c_opt)

        # The transformed points should be left multiplied with w2c_opt, thus premult the inverse of the resd
        ray_o = point_padding(ray_o) @ w2c_ori.mT @ inv_w2c_opt.mT  # B, N, 4 @ B, 4, 4
        ray_d = vector_padding(ray_d) @ w2c_ori.mT @ inv_w2c_opt.mT  # B, N, 4 @ B, 4, 4
        return ray_o[..., :3], ray_d[..., :3]

    def forward(self, ray_o: torch.Tensor, ray_d: torch.Tensor, batch):
        batch = self.forward_cams(batch)
        batch = self.forward_srcs(batch)
        ray_o, ray_d = self.forward_rays(ray_o, ray_d, batch)
        return ray_o, ray_d, batch
