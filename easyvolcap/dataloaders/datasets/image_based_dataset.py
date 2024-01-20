import cv2
import torch
import random
import numpy as np
from functools import lru_cache

from typing import List, Dict, Union
from easyvolcap.engine import cfg, args
from easyvolcap.engine import DATASETS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import DataSplit, pin_memory, to_tensor, as_torch_func
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset


# We have a tricky situation here:
# There are datasets that stores all images in a single folder
# We want to sample closest camera from all those images with ease
# While maintaining the ability to use different kinds of dataloading techniques
# It's essentially a dataset transpose problem
# Maybe the best solution is to just store all images in a separated folder?
# For now, we need to add these dirty supports for
# 1. Distributed training -> only convert latent_index (which might be used for view selection)
# 2. Training dataloading -> there's a definition of latent and view index, so it's quite easy
# 3. Inference (especially gui) dataloading -> no definition of so called view_index -> tricky implementation of selection logic

# For distributed training, should not perform datasharding since all source views are needed
# Or we could implement some dataset based sharding technology cleverly?

@DATASETS.register_module()
class ImageBasedDataset(VolumetricVideoDataset):
    def __init__(self,
                 n_srcs_list: List[int] = [2, 3, 4],  # MARK: repeated global configuration
                 n_srcs_prob: List[int] = [0.2, 0.6, 0.2],  # MARK: repeated global configuration
                 append_gt_prob: float = 0.1,
                 extra_src_pool: int = 1,
                 closest_using_t: bool = False,  # find the closest view using the temporal dimension
                 supply_decoded: bool = False,
                 barebone: bool = False,

                 src_view_sample: List[int] = [0, None, 1],  # use these as input source views
                 force_sparse_view: bool = True, # The user will be responsible for setting up the correct view count

                 **kwargs,
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs)

        self.closest_using_t = closest_using_t
        self.src_view_sample = src_view_sample
        assert not self.closest_using_t or self.frame_sample == [0, None, 1] or force_sparse_view, "Should use default frame_sample [0, None, 1] for ibr dataset with `closest_using_t`. Control sampling through sampler.frame_sample and src_view_sample"
        assert self.view_sample == [0, None, 1] or force_sparse_view, "Should use default view_sample [0, None, 1] for ibr dataset. Control sampling through sampler.view_sample and src_view_sample"
        assert not (self.cache_raw and not supply_decoded), "Will always supply decoded source images when cache_raw is enabled for faster sampling, set cache_raw to False to supply jpeg streams"
        # if self.src_view_sample != [0, None, 1] and self.view_sample != [0, None, 1]: log(red(f'Using `src_view_sample = {self.src_view_sample}` when `view_sample = {self.view_sample}` is not default'))
        # if tar_view_sample != [0, None, 1] and view_sample != [0, None, 1]: log(red(f'Using `src_view_sample = {tar_view_sample}` when `view_sample = {self.view_sample}` is not default'))
        # self.tar_view_sample = tar_view_sample

        # Views are selected and loaded
        # Frames are selected and loaded
        self.load_source_params()

        # Need to build all possible view selections (distance of c2w)
        # - Dot product of v_front - euclidian distance of center
        self.load_source_indices()

        self.n_srcs_list = n_srcs_list if len(n_srcs_list) != 1 or n_srcs_list[0] != 0 else [self.n_views]
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = extra_src_pool
        self.append_gt_prob = append_gt_prob

        # src_inps will come in as decoded bytes instead of jpegs
        self.supply_decoded = supply_decoded
        self.barebone = barebone

    def load_source_params(self):
        # Perform view selection first
        view_inds = self.frame_inds if self.closest_using_t else self.view_inds
        view_inds = torch.arange(0, len(view_inds))
        if len(self.src_view_sample) != 3: view_inds = view_inds[self.src_view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.src_view_sample[0]:self.src_view_sample[1]:self.src_view_sample[2]]  # begin, start, end
        self.src_view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # Controls whether the interpolation is performed on the frame or view dim
        # self.src_view_inds = self.frame_inds[view_inds] if self.closest_using_t else self.view_inds[view_inds]
        # self.src_view_inds = self.frame_inds if self.closest_using_t else self.view_inds

        # For getting the actual data (renaming w2c and K)
        # See easyvolcap/dataloaders/datasets/image_based_inference_dataset
        if self.closest_using_t:  # this checks whether the view selection is performed on the frame or view dim
            self.src_ixts = self.Ks[:, view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[:, view_inds])  # N, L, 4, 4
            self.src_ixts = self.src_ixts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
            self.src_exts = self.src_exts.permute(1, 0, 2, 3)  # L, N, 4, 4 # MARK: transpose
        else:
            self.src_ixts = self.Ks[view_inds]  # N, L, 4, 4
            self.src_exts = affine_padding(self.w2cs[view_inds])  # N, L, 4, 4

    def load_source_indices(self):
        tar_c2ws = self.c2ws.permute(1, 0, 2, 3) if self.closest_using_t else self.c2ws  # MARK: transpose
        src_c2ws = affine_inverse(self.src_exts)
        centers_target = tar_c2ws[..., :3, 3]  # N, L, 3
        centers_source = src_c2ws[..., :3, 3]  # N, L, 3

        # Using distance between centers for camera selection
        sims: torch.Tensor = 1 / (centers_source[None] - centers_target[:, None]).norm(dim=-1)  # N, N, L,

        # Source view index and there similarity
        self.src_sims, self.src_inds = sims.sort(dim=1, descending=True)  # similarity to source views # Target, Source, Latent

    def get_metadata(self, index: dotdict):
        if isinstance(index, dotdict): index, n_srcs = index.index, index.n_srcs
        else: n_srcs = random.choices(self.n_srcs_list, self.n_srcs_prob)[0]

        # Load target view related stuff
        output = VolumetricVideoDataset.get_metadata(self, index)  # target view camera matrices

        # Load target view enerf specific stuff
        # There's this strange convension in ENeRF: ext: 4x4 homogeneous matrix, c2w: 3x4 reduced mat
        output.tar_ext = affine_padding(output.w2c)  # ? renaming things
        output.tar_ixt = output.K  # 3, 3, avoid being modified later

        # Load source view related stuff
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            target_index = output.latent_index
            extra_index = output.view_index
        else:
            target_index = output.view_index
            extra_index = output.latent_index

        # For training, maybe sample the original image
        remove_gt = 1 if random.random() > self.append_gt_prob else 0  # training and random -> exclude gt
        random_ap = self.extra_src_pool  # training -> randomly sample more
        src_inds = self.src_inds[target_index, remove_gt:remove_gt + n_srcs + random_ap, extra_index]  # excluding the target view, 5 inds
        if random_ap: src_inds = torch.as_tensor(random.sample(src_inds.numpy().tolist(), n_srcs))  # S (2, 4)

        output.t_inds = extra_index
        output.meta.t_inds = extra_index
        output.src_exts = self.src_exts[src_inds, extra_index]  # S, 4, 4
        output.src_ixts = self.src_ixts[src_inds, extra_index]  # S, 3, 3
        output.meta.src_exts = output.src_exts
        output.meta.src_ixts = output.src_ixts

        # Other bookkeepings
        src_inds = self.src_view_inds.gather(-1, src_inds)  # S, -> T, S, L -> T, S, L
        output.src_inds = src_inds  # as tensors
        output.meta.src_inds = src_inds  # as tensors

        source_index = src_inds.detach().cpu().numpy().tolist()
        if self.closest_using_t:  # selecting closest view along temporal dimension # MARK: transpose
            latent_index = source_index
            view_index = extra_index
        else:
            latent_index = extra_index
            view_index = source_index

        output = self.get_sources(latent_index, view_index, output)

        return output

    def get_sources(self, latent_index: Union[List[int], int], view_index: Union[List[int], int], output: dotdict):
        if self.split == DataSplit.TRAIN or self.supply_decoded:  # most of the time we asynchronously load images for training, thus no need to decode them using nvjpeg
            rgb, msk, wet, dpt, bkg = zip(*parallel_execution(view_index, latent_index, action=self.get_image, sequential=True))
            output.src_inps = [i.permute(2, 0, 1) for i in rgb]  # for data locality # S, H, W, 3 -> S, 3, H, W
            if msk[0] is not None: output.src_msks = [i.permute(2, 0, 1) for i in msk]  # for data locality # S, H, W, 3 -> S, 3, H, W
            if wet[0] is not None: output.src_wets = [i.permute(2, 0, 1) for i in wet]  # for data locality # S, H, W, 3 -> S, 3, H, W
            if dpt[0] is not None: output.src_dpts = [i.permute(2, 0, 1) for i in dpt]  # for data locality # S, H, W, 3 -> S, 3, H, W
            if bkg[0] is not None: output.src_bkgs = [i.permute(2, 0, 1) for i in bkg]  # for data locality # S, H, W, 3 -> S, 3, H, W
        else:
            im_bytes, mk_bytes, wt_bytes, dp_bytes, bg_bytes = zip(*parallel_execution(view_index, latent_index, action=self.get_image_bytes, sequential=True))
            output.meta.src_inps = im_bytes
            if mk_bytes[0] is not None: output.meta.src_msks = mk_bytes
            if wt_bytes[0] is not None: output.meta.src_wets = wt_bytes
            if dp_bytes[0] is not None: output.meta.src_dpts = dp_bytes
            if bg_bytes[0] is not None: output.meta.src_bkgs = bg_bytes
        return output

    def get_viewer_batch(self, output: dotdict):
        if self.barebone: return VolumetricVideoDataset.get_viewer_batch(self, output)

        # The batch contains H, W, K, R, T, t (time index)
        H, W, K, R, T = output.H, output.W, output.K, output.R, output.T
        n, f, t, bounds = output.n, output.f, output.t, output.bounds
        w2c = torch.cat([R, T], dim=-1)
        c2w = affine_inverse(w2c)

        # Target camera parameters
        output.tar_ixt = K
        output.tar_ext = w2c

        # Source indices
        frame_index = self.t_to_frame(t)
        latent_index = self.frame_to_latent(frame_index)
        view_index = 0  # whatever

        # Update indices, maybe not needed
        output.view_index = view_index
        output.frame_index = frame_index
        output.latent_index = latent_index
        output.meta.view_index = view_index
        output.meta.frame_index = frame_index
        output.meta.latent_index = latent_index

        # Load source view related stuff
        if self.closest_using_t:  # selecting closest view along temporal dimension
            target_index = latent_index
            extra_index = view_index
        else:
            target_index = view_index
            extra_index = latent_index

        center_target = c2w[..., 3]  # 3,
        centers_source = affine_inverse(self.src_exts[:, extra_index])[..., :3, 3]  # N, 3

        sims: torch.Tensor = 1 / (centers_source - center_target).norm(dim=-1).clip(1e-10)  # N,
        src_sims, src_inds = sims.sort(dim=-1, descending=True)  # S,

        n_srcs = self.n_srcs_list[-1]
        src_sims, src_inds = src_sims[:n_srcs], src_inds[:n_srcs]

        # Source camera parameters
        output.t_inds = extra_index  # only for caching the feature extraction results
        output.meta.t_inds = extra_index  # only for caching the feature extraction results
        output.src_exts = self.src_exts[src_inds, extra_index]  # S, 4, 4 # these two are already selected
        output.src_ixts = self.src_ixts[src_inds, extra_index]  # S, 3, 3 # these two are already selected

        # Select the source view indices
        src_inds = self.src_view_inds.gather(-1, src_inds)  # S, -> T, S, L -> T, S, L
        output.src_inds = src_inds  # as tensors
        output.meta.src_inds = src_inds  # as tensors

        # Source images
        source_index = src_inds.detach().cpu().numpy().tolist()
        if self.closest_using_t:  # selecting closest view along temporal dimension
            latent_index = source_index
            view_index = extra_index
        else:
            latent_index = extra_index
            view_index = source_index

        output = self.get_sources(latent_index, view_index, output)

        # Maybe load foreground human prior
        if self.use_objects_priors:
            output = self.get_objects_priors(output)

        # Load bounds
        output.bounds = self.get_bounds(latent_index).clone()  # before inplace operation
        output.bounds[0] = torch.maximum(output.bounds[0], bounds[0])
        output.bounds[1] = torch.minimum(output.bounds[1], bounds[1])
        output.meta.bounds = output.bounds

        # Resize input if needed
        output = self.scale_ixts(output, self.render_ratio)

        # Fill it with zeros in visualizer
        if self.imbound_crop:
            output = self.crop_ixts_bounds(output)

        return output
