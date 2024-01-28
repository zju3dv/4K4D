# Default dataset for loading volumetric videos
# Expected formats: easymocap camera parameters & images folder
# multiview dataset format:
# intri.yml
# extri.yml
# images/
#     00/
#         000000.jpg
#         000001.jpg
#         ...
#     01/
#         000000.jpg
#         000001.jpg
#         ...

# monocular dataset format:
# cameras/
#     00/
#         intri.yml
#         extri.yml
# images/
#     00/
#         000000.jpg
#         000001.jpg
#         ...
# will prepare all camera parameters before the actual data loading
# will perform random samplinng on the rays (either patch or rays)

# The exposed apis should be as little as possible
import os
import cv2  # for undistortion
import torch
import numpy as np
from PIL import Image
from glob import glob
from typing import List
from functools import lru_cache, partial
from torch.utils.data import Dataset, get_worker_info

from easyvolcap.engine import DATASETS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.ray_utils import weighted_sample_rays
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.vhull_utils import hierarchically_carve_vhull
from easyvolcap.utils.cam_utils import average_c2ws, align_c2ws, average_w2cs
from easyvolcap.utils.dist_utils import get_rank, get_world_size, get_distributed
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, torch_inverse_3x3, point_padding
from easyvolcap.utils.bound_utils import get_bound_2d_bound, get_bounds, monotonic_near_far, get_bound_3d_near_far
from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes

cv2.setNumThreads(1)  # MARK: only 1 thread for opencv undistortion, high cpu, not faster


@DATASETS.register_module()
class VolumetricVideoDataset(Dataset):
    def __init__(self,
                 # Dataset intrinsic properties
                 data_root: str,  # this must be configured
                 split: str = DataSplit.TRAIN.name,  # dynamically generated

                 # The frame number & image size should be inferred from the dataset
                 ratio: float = 1.0,  # use original image size
                 center_crop_size: List[int] = [-1, -1],  # center crop image to this size, after resize
                 n_rays: int = 512,  # number of rays to sample from a single image
                 view_sample: List = [0, None, 1],  # begin, end, step
                 frame_sample: List = [0, None, 1],  # begin, end, step
                 correct_pix: bool = True,  # move pixel coordinates to the middle of the pixel
                 use_loaded_time: bool = False,  # use the time provided by the datasets, rare

                 # Other default configurations
                 intri_file: str = 'intri.yml',
                 extri_file: str = 'extri.yml',
                 images_dir: str = 'images',
                 cameras_dir: str = 'cameras',  # only when the camera is moving through time
                 ims_pattern: str = '{frame:06d}.jpg',
                 imsize_overwrite: List[int] = [-1, -1],  # overwrite the image size

                 # Camera alignment
                 use_aligned_cameras: bool = False,
                 avg_using_all: bool = False,  # ok enough for now
                 avg_max_count: int = 100,  # prevent slow center of attention computation
                 init_viewer_index: int = 0,
                 use_avg_init_viewer: bool = False,  # use average camera as initial viewer

                 # Mask related configs
                 masks_dir: str = 'masks',
                 use_masks: bool = False,
                 bkgd_weight: float = 1.0,  # fill bkgd weight with 1.0s
                 imbound_crop: bool = False,
                 immask_crop: bool = False,
                 immask_fill: bool = False,

                 # Depth related configs
                 depths_dir: str = 'depths',
                 use_depths: bool = False,

                 # Human priors # TODO: maybe move these to a different dataset?
                 use_smpls: bool = False,  # use smpls as prior
                 motion_file: str = 'motion.npz',
                 bodymodel_file: str = 'output-smpl-3d/cfg_model.yml',
                 canonical_smpl_file: str = None,

                 # Object priors
                 use_objects_priors: bool = False,  # use foreground prior
                 objects_bounds: List[List[float]] = None,  # manually estimated input objects bounds if there's no masks or smpls

                 # Background priors # TODO: maybe move these to a different dataset?
                 bkgds_dir: str = 'bkgd',  # for those methods who use background images
                 use_bkgds: bool = False,  # use background images

                 # Image preprocessing & formatting
                 use_z_depth: bool = False,
                 dist_opt_K: bool = True,  # use optimized K for undistortion (will crop out black edges), mostly useful for large number of images
                 encode_ext: str = '.jpg',
                 cache_raw: bool = False,
                 ddp_shard_dataset: bool = True,  # for image based rendering, no sharding for now

                 # Visual hull priors # TODO: maybe move to a different module?
                 vhulls_dir: str = 'vhulls',  # for easy reloading of visual hulls
                 use_vhulls: bool = False,  # only usable when loading masks
                 vhull_thresh: float = 0.95,  # 0.9 of all valid cameras sees this point
                 count_thresh: int = 16,  # large common views
                 vhull_padding: float = 0.02,  # smaller, more accurate
                 vhull_voxel_size: float = 0.005,  # smaller -> more accurate
                 vhull_ctof_factor: float = 3.0,  # smaller -> more accurate
                 vhull_thresh_factor: float = 1.0,
                 vhull_count_factor: float = 1.0,
                 force_sparse_vhulls: bool = False,  # use sparse views
                 coarse_discard_masks: bool = False,  # the coarse vhull also requires mask
                 intersect_camera_bounds: bool = True,
                 print_vhull_bounds: bool = False,
                 remove_outlier: bool = True,
                 reload_vhulls: bool = False,  # reload visual hulls to vhulls_dir
                 vhull_only: bool = False,

                 # Volume based config
                 bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 near: float = 0.2,  # lets hope the cameras are not too close
                 far: float = 20,  # TODO: Notify the user about setting up near and far manually or project computed vhull or bounds for better near and far bounds

                 # Dataloading cache config
                 image_cache_maxsize: int = 0,  # 1000 this could be too much
                 preload_images: bool = False,

                 # Dynamically tunable variables
                 render_ratio: float = 1.0,  # might need to resize just before sampling
                 render_center_crop_ratio: float = 1.0,  # might need to center crop just before sampling
                 dist_mask: List[bool] = [1] * 5,
                 skip_loading_images: bool = False,  # for debugging and visualization

                 # Patch sampling related
                 patch_size: List[int] = [-1, -1],  # empty list -> no patch sampling
                 random_crop_size: List[int] = [-1, -1],  # empty list -> no patch sampling

                 # Unused configs, placed here only to remove warnings
                 barebone: bool = True,  # this is always True here
                 supply_decoded: bool = True,  # this is always True here
                 n_srcs_list: List[int] = [8],
                 n_srcs_prob: List[int] = [1.0],
                 append_gt_prob: float = 0.0,
                 temporal_range: List[float] = [0, 1],
                 interp_using_t: bool = False,
                 closest_using_t: bool = False,  # for backwards compatibility
                 force_sparse_view: bool = False,
                 n_render_views: int = 600,
                 extra_src_pool: int = 1,
                 focal_ratio: float = 1.0,
                 src_view_sample: List[int] = [0, None, 1],
                 interp_cfg: dotdict = dotdict(),
                 ):
        # Global dataset config entries
        self.data_root = data_root
        self.intri_file = intri_file
        self.extri_file = extri_file
        self.motion_file = motion_file
        self.bodymodel_file = bodymodel_file
        self.canonical_smpl_file = canonical_smpl_file

        # Camera and alignment configs
        self.avg_using_all = avg_using_all
        self.avg_max_count = avg_max_count
        self.init_viewer_index = init_viewer_index
        self.use_aligned_cameras = use_aligned_cameras
        self.use_avg_init_viewer = use_avg_init_viewer

        # Data and priors directories
        self.cameras_dir = cameras_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.depths_dir = depths_dir
        self.vhulls_dir = vhulls_dir
        self.bkgds_dir = bkgds_dir
        self.ims_pattern = ims_pattern

        # Dynamically tunable configurations
        self.n_rays_shared = torch.as_tensor(n_rays, dtype=torch.long).share_memory_()
        self.patch_size_shared = torch.as_tensor(patch_size, dtype=torch.long).share_memory_()
        self.render_ratio_shared = torch.as_tensor(render_ratio, dtype=torch.float).share_memory_()
        self.random_crop_size_shared = torch.as_tensor(random_crop_size, dtype=torch.float).share_memory_()
        self.render_center_crop_ratio_shared = torch.as_tensor(render_center_crop_ratio, dtype=torch.float).share_memory_()

        # Camera and image selection
        self.frame_sample = frame_sample
        self.view_sample = view_sample
        if self.view_sample[1] is not None: self.n_view_total = self.view_sample[1]
        else: self.n_view_total = len(os.listdir(join(self.data_root, self.images_dir)))  # total number of cameras before filtering
        if self.frame_sample[1] is not None: self.n_frames_total = self.frame_sample[1]
        else: self.n_frames_total = min([len(glob(join(self.data_root, self.images_dir, cam, '*'))) for cam in os.listdir(join(self.data_root, self.images_dir))])  # total number of images before filtering
        self.use_loaded_time = use_loaded_time

        # Rendering and space carving bounds
        self.bounds = torch.as_tensor(bounds, dtype=torch.float)
        self.near = near
        self.far = far

        # Compute needed visual hulls & align all cameras loaded
        self.load_cameras()  # load and normalize all cameras (center lookat, align y axis)
        self.select_cameras()  # select repective cameras to use

        # Load the actual data (as encoded jpeg bytes)
        self.split = DataSplit[split]
        self.ratio = ratio  # could be a float (shared ratio) or a list of floats (should match images)
        self.encode_ext = encode_ext
        self.cache_raw = cache_raw  # use raw pixels to further accelerate training
        self.use_depths = use_depths  # use visual hulls as a prior
        self.use_vhulls = use_vhulls  # use visual hulls as a prior
        self.use_masks = use_masks  # always load mask if using vhulls
        self.use_smpls = use_smpls  # use smpls as a prior
        self.use_bkgds = use_bkgds  # use background images as a prior
        self.ddp_shard_dataset = ddp_shard_dataset  # shard the dataset between DDP processes
        self.imsize_overwrite = imsize_overwrite  # overwrite loaded image sizes (for enerf)
        self.immask_crop = immask_crop  # maybe crop stored jpeg bytes
        self.immask_fill = immask_fill  # maybe fill stored jpeg bytes
        self.center_crop_size = center_crop_size  # center crop size

        # Distorsion related
        self.dist_mask = dist_mask  # ignore some of the camera parameters
        self.dist_opt_K = dist_opt_K

        # Visual hull computation configuration
        self.vhull_thresh = vhull_thresh
        self.count_thresh = count_thresh
        self.reload_vhulls = reload_vhulls
        self.vhull_padding = vhull_padding
        self.vhull_voxel_size = vhull_voxel_size
        self.vhull_ctof_factor = vhull_ctof_factor
        self.vhull_count_factor = vhull_count_factor
        self.vhull_thresh_factor = vhull_thresh_factor
        self.force_sparse_vhulls = force_sparse_vhulls
        self.coarse_discard_masks = coarse_discard_masks
        self.intersect_camera_bounds = intersect_camera_bounds
        self.print_vhull_bounds = print_vhull_bounds
        self.remove_outlier = remove_outlier
        self.vhull_only = vhull_only

        # Foreground objects prior configuration
        self.use_objects_priors = use_objects_priors
        self.objects_bounds = objects_bounds

        self.load_paths()  # load image files into self.ims
        try:
            if self.use_vhulls and not self.reload_vhulls:
                self.load_vhulls()
                if self.vhull_only:
                    exit(0)
        except:
            assert not (self.use_vhulls and skip_loading_images), 'Visual hull hasn\'t been prepared yet, rerun without `skip_loading_images` once'
            stop_prog()
            start_prog()  # clean up residual progress bar
            pass  # silently error out if no visual hull is found here
        if not skip_loading_images:
            self.load_bytes()  # load image bytes (also load vhulls)
        if self.use_smpls:
            self.load_smpls()  # load smpls (if needed, branch inside)
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/

        # Other entries not needed for initialization
        self.imbound_crop = imbound_crop
        self.correct_pix = correct_pix
        self.use_z_depth = use_z_depth
        self.bkgd_weight = bkgd_weight

        # Cache related config
        self.image_cache_maxsize = image_cache_maxsize  # make this smaller to avoid oom
        if image_cache_maxsize:
            self.get_image = lru_cache(image_cache_maxsize)(self.get_image)
            if preload_images:
                pbar = tqdm(total=len(self), desc=f'Preloading raw imgs for {blue(self.data_root)} {magenta(self.split.name)}')
                for v in range(self.n_views):
                    for l in range(self.n_latents):
                        self.get_image(v, l)
                        pbar.update()

        # Maybe used in other places
        self.barebone = barebone
        self.supply_decoded = supply_decoded
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob
        self.append_gt_prob = append_gt_prob
        self.interp_using_t = interp_using_t
        self.closest_using_t = closest_using_t
        self.force_sparse_view = force_sparse_view
        self.n_render_views = n_render_views
        self.extra_src_pool = extra_src_pool
        self.src_view_sample = src_view_sample
        self.interp_cfg = interp_cfg
        self.temporal_range = temporal_range
        self.focal_ratio = focal_ratio

    @property
    def render_ratio(self):
        return self.render_ratio_shared

    @render_ratio.setter
    def render_ratio(self, value: torch.Tensor):
        self.render_ratio_shared.copy_(torch.as_tensor(value, dtype=self.render_ratio_shared.dtype))  # all values will be changed to this

    @property
    def render_center_crop_ratio(self):
        return self.render_center_crop_ratio_shared

    @render_center_crop_ratio.setter
    def render_center_crop_ratio(self, value: torch.Tensor):
        self.render_center_crop_ratio_shared.copy_(torch.as_tensor(value, dtype=self.render_center_crop_ratio_shared.dtype))  # all values will be changed to this

    @property
    def n_rays(self):
        return self.n_rays_shared

    @n_rays.setter
    def n_rays(self, value: torch.Tensor):
        self.n_rays_shared.copy_(torch.as_tensor(value, dtype=self.n_rays_shared.dtype))  # all values will be changed to this

    @property
    def patch_size(self):
        return self.patch_size_shared

    @patch_size.setter
    def patch_size(self, value: torch.Tensor):
        self.patch_size_shared.copy_(torch.as_tensor(value, dtype=self.patch_size_shared.dtype))  # all values will be changed to this

    @property
    def random_crop_size(self):
        return self.random_crop_size_shared

    @random_crop_size.setter
    def random_crop_size(self, value: torch.Tensor):
        self.random_crop_size_shared.copy_(torch.as_tensor(value, dtype=self.random_crop_size_shared.dtype))  # all values will be changed to this

    def load_paths(self):
        # Load image related stuff for reading from disk later
        # If number of images in folder does not match, here we'll get an error
        ims = [[join(self.data_root, self.images_dir, cam, self.ims_pattern.format(frame=i)) for i in range(self.n_frames_total)] for cam in self.camera_names]
        if not exists(ims[0][0]):
            ims = [[i.replace('.' + self.ims_pattern.split('.')[-1], '.JPG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.JPG', '.png') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.png', '.PNG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [sorted(glob(join(self.data_root, self.images_dir, cam, '*')))[:self.n_frames_total] for cam in self.camera_names]
        ims = [np.asarray(ims[i])[:min([len(i) for i in ims])] for i in range(len(ims))]  # deal with the fact that some weird dataset has different number of images
        self.ims = np.asarray(ims)  # V, N
        self.ims_dir = join(*split(dirname(self.ims[0, 0]))[:-1])  # logging only

        # TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        # MARK: Names stored as np.ndarray
        inds = np.arange(self.ims.shape[-1])
        if len(self.frame_sample) != 3: inds = inds[self.frame_sample]
        else: inds = inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.ims = self.ims[..., inds]  # these paths are later used for reading images from disk

        # MARK: Handle explicit dataset sharding here
        if self.split == DataSplit.TRAIN:
            self.rank = get_rank() if self.ddp_shard_dataset else 0  # 0 for non ddp
            self.num_replicas = get_world_size() if self.ddp_shard_dataset else 1  # 1 for non ddp
            self.num_samples = np.ceil(self.n_latents / self.num_replicas)  # total number of samples

            # Shard the image names
            self.ims = self.ims[:, self.rank::self.num_replicas]  # shard on number of frames

            # Shard the cameras
            self.cameras = dotdict({k: self.cameras[k][self.rank::self.num_replicas] for k in self.camera_names})  # reloading
            self.Ks = self.Ks[:, self.rank::self.num_replicas]
            self.Rs = self.Rs[:, self.rank::self.num_replicas]
            self.Ts = self.Ts[:, self.rank::self.num_replicas]
            self.Ds = self.Ds[:, self.rank::self.num_replicas]
            self.ts = self.ts[:, self.rank::self.num_replicas]  # controlled by use_loaded_time, default false, using computed t from frame_sample
            self.Cs = self.Cs[:, self.rank::self.num_replicas]
            self.w2cs = self.w2cs[:, self.rank::self.num_replicas]
            self.c2ws = self.c2ws[:, self.rank::self.num_replicas]

        # Mask path preparation
        if self.use_masks:
            self.mks = np.asarray([im.replace(self.images_dir, self.masks_dir) for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.png', '.jpg') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('.jpg', '.png') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):  # Two types of commonly used mask directories
                self.mks = np.asarray([mk.replace(self.masks_dir, 'masks') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'mask') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            if not exists(self.mks[0, 0]):
                self.mks = np.asarray([mk.replace('masks', 'msk') for mk in self.mks.ravel()]).reshape(self.mks.shape)
            self.mks_dir = join(*split(dirname(self.mks[0, 0]))[:-1])

        # Depth image path preparation
        if self.use_depths:
            self.dps = np.asarray([im.replace(self.images_dir, self.depths_dir).replace('.jpg', '.exr').replace('.png', '.exr') for im in self.ims.ravel()]).reshape(self.ims.shape)
            if not exists(self.dps[0, 0]):
                self.dps = np.asarray([dp.replace('.exr', 'exr') for dp in self.dps.ravel()]).reshape(self.dps.shape)
            self.dps_dir = join(*split(dirname(self.dps[0, 0]))[:-1])  # logging only

        # Background image path preparation
        if self.use_bkgds:
            self.bgs = np.asarray([join(self.data_root, self.bkgds_dir, f'{cam}.jpg') for cam in self.camera_names])  # V,
            if not os.path.exists(self.bgs[0]):
                self.bgs = np.asarray([bg.replace('.jpg', '.png') for bg in self.bgs])
            self.bgs_dir = join(*split(dirname(self.bgs[0]))[:-1])  # logging only

    def load_bytes(self):
        # Camera distortions are only applied on the ground truth image, the rendering model does not include these
        # And unlike intrinsic parameters, it has no direct dependency on the size of the loaded image, thus we directly process them here
        dist_mask = torch.as_tensor(self.dist_mask)
        self.Ds = self.Ds.view(*self.Ds.shape[:2], 5) * dist_mask  # some of the distortion parameters might need some manual massaging

        # Need to convert to a tight data structure for access
        ori_Ks = self.Ks
        ori_Ds = self.Ds
        # msk_Ds = ori_Ds.clone()  # this is a DNA-Rendering special
        # msk_Ds[..., -1] = 0.0  # only use the first 4 distortion parameters for mask undistortion
        # msk_Ds = torch.zeros_like(ori_Ds) # avoid bad distortion params
        ratio = self.imsize_overwrite if self.imsize_overwrite[0] > 0 else self.ratio  # maybe force size, or maybe use ratio to resize
        if self.use_masks:
            self.mks_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.mks, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading mask bytes for {blue(self.mks_dir)} {magenta(self.split.name)}',
                                             decode_flag=cv2.IMREAD_GRAYSCALE, dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)  # will for a grayscale read from bytes
            self.Ks = torch.as_tensor(self.Ks)
            self.Hs = torch.as_tensor(self.Hs)
            self.Ws = torch.as_tensor(self.Ws)

        # Maybe compute visual hulls after loading the dataset
        if self.use_vhulls and not hasattr(self, 'vhulls'):
            self.load_vhulls()  # before cropping the mask (we need all the information we can get for visual hulls)
            if self.vhull_only:
                exit(0)

        # Maybe load background images here
        if self.use_bkgds:
            self.bgs_bytes, _, _, _ = \
                load_resize_undist_ims_bytes(self.bgs, ori_Ks[:, 0].numpy(), ori_Ds[:, 0].numpy(), ratio, self.center_crop_size,
                                             f'Loading bkgd bytes for {blue(self.bgs_dir)} {magenta(self.split.name)}',
                                             dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        # Maybe load depth images here, using HDR
        if self.use_depths:  # TODO: implement HDR loading
            self.dps_bytes, self.Ks, self.Hs, self.Ws = \
                load_resize_undist_ims_bytes(self.dps, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                             f'Loading dpts bytes for {blue(self.dps_dir)} {magenta(self.split.name)}',
                                             decode_flag=cv2.IMREAD_UNCHANGED, dist_opt_K=self.dist_opt_K, encode_ext='.exr')  # will for a grayscale read from bytes

        # Image pre cacheing (from disk to memory)
        self.ims_bytes, self.Ks, self.Hs, self.Ws = \
            load_resize_undist_ims_bytes(self.ims, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                         f'Loading imgs bytes for {blue(self.ims_dir)} {magenta(self.split.name)}',
                                         dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        self.Ks = torch.as_tensor(self.Ks)
        self.Hs = torch.as_tensor(self.Hs)
        self.Ws = torch.as_tensor(self.Ws)

        # Precrop image to bytes
        if self.immask_crop:  # a little bit wasteful but acceptable for now
            self.orig_hs, self.orig_ws = self.Hs, self.Ws
            bounds = [self.get_bounds(i) for i in range(self.n_latents)]  # N, 2, 3
            bounds = torch.stack(bounds)[None].repeat(self.n_views, 1, 1, 1)  # V, N, 2, 3
            self.ims_bytes, self.mks_bytes, self.Ks, self.Hs, self.Ws, self.crop_xs, self.crop_ys = \
                decode_crop_fill_ims_bytes(self.ims_bytes, self.mks_bytes, self.Ks.numpy(), self.Rs.numpy(), self.Ts.numpy(), bounds.numpy(), f'Cropping msks imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)
            if hasattr(self, 'dps_bytes'): self.dps_bytes, self.mks_bytes, self.Ks, self.Hs, self.Ws, self.crop_xs, self.crop_ys = \
                decode_crop_fill_ims_bytes(self.dps_bytes, self.mks_bytes, self.Ks.numpy(), self.Rs.numpy(), self.Ts.numpy(), bounds.numpy(), f'Cropping msks dpts for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=['.exr', self.encode_ext])
            self.corp_xs = torch.as_tensor(self.crop_xs)
            self.corp_ys = torch.as_tensor(self.crop_ys)
            self.Ks = torch.as_tensor(self.Ks)
            self.Hs = torch.as_tensor(self.Hs)
            self.Ws = torch.as_tensor(self.Ws)

        # Only fill the background regions
        if not self.immask_crop and self.immask_fill:  # a little bit wasteful but acceptable for now
            self.ims_bytes = decode_fill_ims_bytes(self.ims_bytes, self.mks_bytes, f'Filling msks imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext=self.encode_ext)
            if hasattr(self, 'dps_bytes'): self.dps_bytes = decode_fill_ims_bytes(self.dps_bytes, self.mks_bytes, f'Filling dpts imgs for {blue(self.data_root)} {magenta(self.split.name)}', encode_ext='.exr')

        # To make memory access faster, store raw floats in memory
        if self.cache_raw:
            self.ims_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.ims_bytes, desc=f'Caching imgs for {blue(self.data_root)} {magenta(self.split.name)}')])  # High mem usage
            if hasattr(self, 'mks_bytes'): self.mks_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.mks_bytes, desc=f'Caching mks for {blue(self.data_root)} {magenta(self.split.name)}')])
            if hasattr(self, 'dps_bytes'): self.dps_bytes = to_tensor([load_image_from_bytes(x, normalize=False) for x in tqdm(self.dps_bytes, desc=f'Caching dps for {blue(self.data_root)} {magenta(self.split.name)}')])
            if hasattr(self, 'bgs_bytes'): self.bgs_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.bgs_bytes, desc=f'Caching bgs for {blue(self.data_root)} {magenta(self.split.name)}')])
        else:
            # Avoid splitting memory for bytes objects
            self.ims_bytes = UnstructuredTensors(self.ims_bytes)
            if hasattr(self, 'mks_bytes'): self.mks_bytes = UnstructuredTensors(self.mks_bytes)
            if hasattr(self, 'dps_bytes'): self.dps_bytes = UnstructuredTensors(self.dps_bytes)
            if hasattr(self, 'bgs_bytes'): self.bgs_bytes = UnstructuredTensors(self.bgs_bytes)

    def load_vhulls(self):

        # Implement a visual hull carving method
        # This should be fast enough for preprocessing
        # We assume all camera centers as a outside bound for the visual hull
        # And we assume all cameras has been aligned to look at roughly the origin
        def carve_using_bytes(H, W, K, R, T, latent_index):
            bytes = [self.mks_bytes[i * self.n_latents + latent_index] for i in range(len(H))]  # get mask bytes of this frame
            if bytes[0].ndim != 3:
                msk = parallel_execution(bytes, normalize=True, action=load_image_from_bytes, sequential=True)
            else:
                msk = bytes
            msks = to_tensor(msk)

            # Fill blank canvas for each mask
            # It should be OK to use black images without resizing since we're performing grid_sampling
            N = len(msks)
            H_max = max([msk.shape[-3] for msk in msks])
            W_max = max([msk.shape[-2] for msk in msks])
            msks_full = H.new_zeros(N, H_max, W_max, 1, dtype=torch.float)
            for i, (h, w, msk) in enumerate(zip(H, W, msks)):
                msks_full[i, :h, :w, :] = msk  # fill
            msks = msks_full  # N, H, W, 1

            # Actual carving
            inputs = H, W, K, R, T, msks, self.bounds  # this is the starting bounds
            inputs = to_cuda(inputs)
            vhulls, bounds, valid, inds = hierarchically_carve_vhull(*inputs,
                                                                     padding=self.vhull_padding,
                                                                     voxel_size=self.vhull_voxel_size,
                                                                     ctof_factor=self.vhull_ctof_factor,
                                                                     vhull_thresh=self.vhull_thresh,
                                                                     count_thresh=self.count_thresh,
                                                                     vhull_thresh_factor=self.vhull_thresh_factor,
                                                                     vhull_count_factor=self.vhull_count_factor,
                                                                     coarse_discard_masks=self.coarse_discard_masks,
                                                                     intersect_camera_bounds=self.intersect_camera_bounds,
                                                                     remove_outlier=self.remove_outlier,
                                                                     )
            vhulls, bounds, valid, inds = to_cpu([vhulls, bounds, valid, inds])  # always blocking

            return vhulls, bounds, valid, inds

        # Need to consider supporting space carving of inference dataset where only as subset of the images will be preloaded
        # Otherwise there's a duplicated loading -> double memory usage during training (with inference validation)

        # Load visual hulls from files if exists
        self.vhs = np.asarray([
            join(
                split(split(im)[0])[0].replace(self.images_dir, self.vhulls_dir),
                split(im)[-1].replace('.png', '.ply').replace('.jpg', '.ply')
            )
            for im in self.ims[0]
        ])
        if not exists(self.vhs[0]):  # Two commonly used visual hull directories
            self.vhs = [vh.replace(self.vhulls_dir, 'surfs') for vh in self.vhs]
        if not exists(self.vhs[0]):
            self.vhs = [vh.replace('surfs', 'vhulls') for vh in self.vhs]

        Hs = [_ for _ in self.Hs.movedim(0, 1)]  # F, V,
        Ws = [_ for _ in self.Ws.movedim(0, 1)]
        Ks = [_ for _ in self.Ks.movedim(0, 1)]  # F, V, 3, 3
        Rs = [_ for _ in self.Rs.movedim(0, 1)]
        Ts = [_ for _ in self.Ts.movedim(0, 1)]

        vhulls_bounds = []
        for latent_index, (vh, H, W, K, R, T) in enumerate(tqdm(zip(self.vhs, Hs, Ws, Ks, Rs, Ts), total=len(Hs), desc=f'Preparing vhulls {blue(join(self.data_root, self.vhulls_dir))} {magenta(self.split.name)}')):

            # MARK: Always store point clouds in world coordinates
            if not exists(vh) or self.reload_vhulls:
                vhulls, bounds, valid, inds = carve_using_bytes(H, W, K, R, T, latent_index)
                if self.use_aligned_cameras:
                    mat = affine_padding(self.c2w_avg)  # 4, 4
                    world_vhulls = (point_padding(vhulls) @ mat.mT)  # homo
                    world_vhulls = world_vhulls[..., :3] / world_vhulls[..., 3:]
                else:
                    world_vhulls = vhulls
                export_pts(world_vhulls, filename=vh)
            else:
                world_vhulls = torch.as_tensor(load_pts(vh)[0])  # N, 3; 2, 3
                if self.use_aligned_cameras:
                    mat = affine_inverse(affine_padding(self.c2w_avg))  # 4, 4
                    vhulls = (point_padding(world_vhulls) @ mat.mT)[..., :3]  # homo
                else:
                    vhulls = world_vhulls
                bounds = get_bounds(vhulls[None])[0]

            vhulls_bounds.append([vhulls, bounds])
        vhulls, bounds = zip(*vhulls_bounds)

        # Everything just for a bounding box...
        self.vhulls = vhulls  # F, N, 3, differnt shape, # MARK: cannot stack this
        self.vhull_bounds = torch.stack(bounds)  # F, 2, 3
        if self.print_vhull_bounds:
            log(magenta(f'Individual visual hull bounds of'))
            for i in range(len(self.vhull_bounds)):
                log(f'{i}:', line(self.vhull_bounds[i]))
            log(magenta(f'Accumulated vhull bound of frames: {self.frame_sample}: '),
                line(torch.stack([self.vhull_bounds.min(dim=0)[0][0],
                                  self.vhull_bounds.max(dim=0)[0][1]])))

    def load_smpls(self):
        # Need to add or complete __getitem__ utils function if smpl paramaters other than bound are needed

        # Import easymocap body model for type annotation
        from easyvolcap.utils.data_utils import get_rigid_transform, load_dotdict

        # Load smpl body model
        # self.bodymodel: SMPLHModel = load_bodymodel(self.data_root, self.bodymodel_file)

        # Load smpl parameters, assume only one person now, TODO: support multiple people
        self.motion = to_tensor(load_dotdict(join(self.data_root, self.motion_file)))

        def get_lbs_params(i):
            poses = self.motion.poses[i][None]  # 1, J * 3
            shapes = self.motion.shapes[i][None]  # 1, S
            Rh = self.motion.Rh[i][None]  # 1, 3,
            Th = self.motion.Th[i][None]  # 1, 3,

            # adjust the smpl pose according to the aligned camera
            if self.use_aligned_cameras:
                R = torch.from_numpy(cv2.Rodrigues(Rh[0].numpy())[0])  # 3, 3
                Rt = torch.cat([R, Th[0].view(3, 1)], dim=1)  # 3, 4
                Rt = (affine_inverse(affine_padding(self.c2w_avg)) @ affine_padding(Rt))[:3, :]  # 3, 4
                Rh = torch.from_numpy(cv2.Rodrigues(Rt[:, :-1].numpy())[0]).view(1, 3)  # 1, 3
                Th = Rt[:, -1].view(1, 3)  # 1, 3

            return poses, shapes, Rh, Th

        smpl_lbs = []
        for i in tqdm(self.frame_inds, desc=f'Loading smpl parameters'):
            poses, shapes, Rh, Th = get_lbs_params(i)
            smpl_lbs.append([poses, shapes, Rh, Th])
        poses, shapes, Rh, Th = zip(*smpl_lbs)
        self.smpl_motions = dotdict()
        self.smpl_motions.poses = torch.cat(poses)
        self.smpl_motions.shapes = torch.cat(shapes)
        self.smpl_motions.Rh = torch.cat(Rh)
        self.smpl_motions.Th = torch.cat(Th)

    def load_cameras(self):
        # Load camera related stuff like image list and intri, extri.
        # Determine whether it is a monocular dataset or multiview dataset based on the existence of root `extri.yml` or `intri.yml`
        # Multiview dataset loading, need to expand, will have redundant information
        if exists(join(self.data_root, self.intri_file)) and exists(join(self.data_root, self.extri_file)):
            self.cameras = read_camera(join(self.data_root, self.intri_file), join(self.data_root, self.extri_file))
            self.camera_names = np.asarray(sorted(list(self.cameras.keys())))  # NOTE: sorting camera names
            self.cameras = dotdict({k: [self.cameras[k] for i in range(self.n_frames_total)] for k in self.camera_names})
            # TODO: Handle avg processing

        # Monocular dataset loading, each camera has a separate folder
        elif exists(join(self.data_root, self.cameras_dir)):
            self.camera_names = np.asarray(sorted(os.listdir(join(self.data_root, self.cameras_dir))))  # NOTE: sorting here is very important!
            self.cameras = dotdict({
                k: [v[1] for v in sorted(
                    read_camera(join(self.data_root, self.cameras_dir, k, self.intri_file),
                                join(self.data_root, self.cameras_dir, k, self.extri_file)).items()
                )] for k in self.camera_names
            })
            # TODO: Handle avg export and loading for such monocular dataset
        else:
            raise NotImplementedError(f'Could not find {{{self.intri_file},{self.extri_file}}} or {self.cameras_dir} directory in {self.data_root}, check your dataset configuration')

        # Expectation:
        # self.camera_names: a list containing all camera names
        # self.cameras: a mapping from camera names to a list of camera objects
        # (every element in list is an actual camera for that particular view and frame)
        # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
        self.Hs = torch.as_tensor([[cam.H for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ws = torch.as_tensor([[cam.W for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ks = torch.as_tensor([[cam.K for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Rs = torch.as_tensor([[cam.R for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Ts = torch.as_tensor([[cam.T for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 1
        self.Ds = torch.as_tensor([[cam.D for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 1, 5
        self.ts = torch.as_tensor([[cam.t for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.ns = torch.as_tensor([[cam.n for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.fs = torch.as_tensor([[cam.f for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Cs = -self.Rs.mT @ self.Ts  # V, F, 3, 1
        self.w2cs = torch.cat([self.Rs, self.Ts], dim=-1)  # V, F, 3, 4
        self.c2ws = affine_inverse(self.w2cs)  # V, F, 3, 4
        self.ns, self.fs = monotonic_near_far(self.ns, self.fs, torch.as_tensor(self.near, dtype=torch.float), torch.as_tensor(self.far, dtype=torch.float))
        self.near, self.far = max(self.near, self.ns.min()), min(self.far, self.fs.max())

        # Move cameras to the center of the frame (!: intrusive)
        if self.use_aligned_cameras:
            self.align_cameras()

        # Compute and set the initial view direction for viewer
        if self.use_avg_init_viewer:
            w2c_avg = as_torch_func(average_w2cs)(self.w2cs[:, 0].numpy())
            self.Rv, self.Tv = w2c_avg[:3, :3], w2c_avg[:3, 3:]
        else:
            self.Rv, self.Tv = self.Rs[self.init_viewer_index, 0], self.Ts[self.init_viewer_index, 0]

    def align_cameras(self):
        sh = self.c2ws.shape  # V, F, 3, 4
        self.c2ws = self.c2ws.view((-1,) + sh[-2:])  # V*F, 3, 4

        if self.avg_using_all:
            stride = max(len(self.c2ws) // self.avg_max_count, 1)
            inds = torch.arange(len(self.c2ws))[::stride][:self.avg_max_count]
            c2w_avg = as_torch_func(average_c2ws)(self.c2ws[inds])  # V*F, 3, 4, # !: HEAVY
        else:
            c2w_avg = as_torch_func(average_c2ws)(self.c2ws.view(sh)[:, 0])  # V, 3, 4
        self.c2w_avg = c2w_avg

        self.c2ws = (affine_inverse(affine_padding(self.c2w_avg))[None] @ affine_padding(self.c2ws))[..., :3, :]  # 1, 4, 4 @ V*F, 4, 4 -> V*F, 3, 4
        self.w2cs = affine_inverse(self.c2ws)  # V*F, 3, 4
        self.c2ws = self.c2ws.view(sh)
        self.w2cs = self.w2cs.view(sh)

        self.Rs = self.w2cs[..., :-1]
        self.Ts = self.w2cs[..., -1:]
        self.Cs = self.c2ws[..., -1:]  # updated camera center

    def select_cameras(self):
        # Only retrain needed
        # Perform view selection first
        view_inds = torch.arange(self.Ks.shape[0])
        if len(self.view_sample) != 3: view_inds = view_inds[self.view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.view_sample[0]:self.view_sample[1]:self.view_sample[2]]  # begin, start, end
        self.view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # Perform frame selection next
        frame_inds = torch.arange(self.Ks.shape[1])
        if len(self.frame_sample) != 3: frame_inds = frame_inds[self.frame_sample]
        else: frame_inds = frame_inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.frame_inds = frame_inds  # used by `load_smpls()`
        if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # NOTE: if view_inds == [0,] in monocular dataset or whatever case, type(`self.camera_names[view_inds]`) == str, not a list of str
        self.camera_names = np.asarray([self.camera_names[view] for view in view_inds])  # this is what the b, e, s means
        self.cameras = dotdict({k: [self.cameras[k][int(i)] for i in frame_inds] for k in self.camera_names})  # reloading
        self.Hs = self.Hs[view_inds][:, frame_inds]
        self.Ws = self.Ws[view_inds][:, frame_inds]
        self.Ks = self.Ks[view_inds][:, frame_inds]
        self.Rs = self.Rs[view_inds][:, frame_inds]
        self.Ts = self.Ts[view_inds][:, frame_inds]
        self.Ds = self.Ds[view_inds][:, frame_inds]
        self.ts = self.ts[view_inds][:, frame_inds]
        self.Cs = self.Cs[view_inds][:, frame_inds]
        self.w2cs = self.w2cs[view_inds][:, frame_inds]
        self.c2ws = self.c2ws[view_inds][:, frame_inds]

    def physical_to_virtual(self, latent_index):
        if self.split != DataSplit.TRAIN:
            return latent_index
        return latent_index * self.num_replicas + self.rank

    def virtual_to_physical(self, latent_index):
        if self.split != DataSplit.TRAIN:
            return latent_index
        return (latent_index - self.rank) // self.num_replicas

    # NOTE: everything beginning with get are utilities for __getitem__
    # NOTE: coding convension are preceded with "NOTE"
    def get_indices(self, index):
        # These indices are relative to the processed dataset
        view_index, latent_index = index // self.n_latents, index % self.n_latents
        latent_index = self.physical_to_virtual(latent_index)

        if len(self.view_sample) != 3: camera_index = self.view_sample[view_index]
        else: camera_index = view_index * self.view_sample[2] + self.view_sample[0]

        if len(self.frame_sample) != 3: frame_index = self.frame_sample[latent_index]
        else: frame_index = latent_index * self.frame_sample[2] + self.frame_sample[0]

        return view_index, latent_index, camera_index, frame_index

    def get_image_bytes(self, view_index: int, latent_index: int):
        latent_index = self.virtual_to_physical(latent_index)
        im_bytes = self.ims_bytes[view_index * self.n_latents + latent_index]  # MARK: no fancy indexing
        if self.use_masks:
            mk_bytes = self.mks_bytes[view_index * self.n_latents + latent_index]
            wt_bytes = mk_bytes.clone()
        else:
            mk_bytes, wt_bytes = None, None

        if self.use_depths:
            dp_bytes = self.dps_bytes[view_index * self.n_latents + latent_index]
        else:
            dp_bytes = None

        if self.use_bkgds:
            bg_bytes = self.bgs_bytes[view_index]
        else:
            bg_bytes = None

        return im_bytes, mk_bytes, wt_bytes, dp_bytes, bg_bytes

    def get_image(self, view_index: int, latent_index: int):
        # Load bytes (rgb, msk, wet, bg)
        im_bytes, mk_bytes, wt_bytes, dp_bytes, bg_bytes = self.get_image_bytes(view_index, latent_index)
        rgb, msk, wet, dpt, bkg = None, None, None, None, None

        # Load image from bytes
        if self.cache_raw:
            rgb = torch.as_tensor(im_bytes)
        else:
            rgb = torch.as_tensor(load_image_from_bytes(im_bytes, normalize=True))  # 4-5ms for 400 * 592 jpeg, sooo slow

        # Load mask from bytes
        if mk_bytes is not None:
            if self.cache_raw:
                msk = torch.as_tensor(mk_bytes)
            else:
                msk = torch.as_tensor(load_image_from_bytes(mk_bytes, normalize=True)[..., :1])
        else:
            msk = torch.ones_like(rgb[..., -1:])

        # Load sampling weights from bytes
        if wt_bytes is not None:
            if self.cache_raw:
                wet = torch.as_tensor(wt_bytes)
            else:
                wet = torch.as_tensor(load_image_from_bytes(wt_bytes, normalize=True)[..., :1])
        else:
            wet = msk.clone()
        wet[msk < self.bkgd_weight] = self.bkgd_weight

        # Load depth from bytes
        if dp_bytes is not None:
            if self.cache_raw:
                dpt = torch.as_tensor(dp_bytes)
            else:
                dpt = torch.as_tensor(load_image_from_bytes(dp_bytes, normalize=False)[..., :1])  # readin as is

        # Load background image from bytes
        if bg_bytes is not None:
            bg_bytes = self.bgs_bytes[view_index]
            if self.cache_raw:
                bkg = torch.as_tensor(bg_bytes)
            else:
                bkg = torch.as_tensor(load_image_from_bytes(bg_bytes, normalize=True))
        return rgb, msk, wet, dpt, bkg

    def get_camera_params(self, view_index, latent_index):
        latent_index = self.virtual_to_physical(latent_index)
        w2c, c2w = self.w2cs[view_index][latent_index], self.c2ws[view_index][latent_index]
        R, T = self.Rs[view_index][latent_index], self.Ts[view_index][latent_index]  # 4, 4; 3, 3; 3, 1; 5, 1
        n, f = self.ns[view_index][latent_index], self.fs[view_index][latent_index]  # 1, 1
        t = self.ts[view_index][latent_index]  # 1

        # These might be invalid
        H, W, K = self.Hs[view_index][latent_index], self.Ws[view_index][latent_index], self.Ks[view_index][latent_index]
        return w2c, c2w, R, T, H, W, K, n, f, t

    def get_bounds(self, latent_index):
        latent_index = self.virtual_to_physical(latent_index)
        bounds = self.vhull_bounds[latent_index] if hasattr(self, 'vhull_bounds') else self.bounds
        bounds = bounds.clone()  # always copy before inplace operation
        bounds[0] = torch.maximum(bounds[0], self.bounds[0])
        bounds[1] = torch.minimum(bounds[1], self.bounds[1])
        return bounds

    def get_smpl_motions(self, latent_index):
        latent_index = self.virtual_to_physical(latent_index)
        smpl_poses = self.smpl_motions.poses[latent_index] if hasattr(self, 'smpl_motions') else None
        smpl_shapes = self.smpl_motions.shapes[latent_index]
        smpl_Rh = self.smpl_motions.Rh[latent_index]
        smpl_Th = self.smpl_motions.Th[latent_index]
        return smpl_poses, smpl_shapes, smpl_Rh, smpl_Th

    def get_objects_bounds(self, latent_index):
        latent_index = self.virtual_to_physical(latent_index)
        if self.use_vhulls: bounds = self.vhull_bounds[latent_index]  # 2, 3
        # TODO: check the current SMPL prior implementation, it seems there's no SMPL bounds for now
        elif self.use_smpls: raise NotImplementedError(f'No SMPL bounds for now')
        elif self.objects_bounds is not None: bounds = torch.as_tensor(self.objects_bounds, dtype=torch.float)  # 2, 3
        else: raise NotImplementedError(f'You must provide either vhulls or smpls or objects_bounds')
        return bounds

    def get_objects_priors(self, output: dotdict):
        latent_index = output.meta.latent_index
        H, W, K, R, T = output.H, output.W, output.K, output.R, output.T

        # TODO: add vhulls or SMPL prior for multiple object priors supporting
        bounds = self.get_objects_bounds(latent_index)
        x, y, w, h = get_bound_2d_bound(bounds, K, R, T, H, W, pad=0)

        # Make the height and width of the bounding box to multiply of 32
        # Adjust the x and y coordinates of the bounding box to make it centered and do not exceed the image size
        H, W = H if isinstance(H, int) else H.item(), W if isinstance(W, int) else W.item()
        x, y, w_orig, h_orig = x.item(), y.item(), w.item(), h.item()
        # Default use `ceil()`, but this may cause h > H at low-resolution, so we use `floor()` instead
        w, h = np.ceil(w_orig / 32) * 32, np.ceil(h_orig / 32) * 32
        if w > W or h > H: w, h = np.floor(w_orig / 32) * 32, np.floor(h_orig / 32) * 32
        x, y = np.clip([x - (w - w_orig) // 2, y - (h - h_orig) // 2], 0, [W - w, H - h])
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Get the near and far depth of the 3d bounding box
        near, far = get_bound_3d_near_far(bounds, R, T)
        objects_bounds, objects_xywh, objects_n, objects_f = [], [], [], []
        objects_bounds.append(bounds)
        objects_xywh.append(torch.tensor([x, y, w, h], dtype=torch.int))
        objects_n.append(near)
        objects_f.append(far)

        meta = dotdict()
        meta.objects_bounds = torch.stack(to_tensor(objects_bounds), dim=0)  # (Nf, 2, 3)
        meta.objects_xywh = torch.stack(objects_xywh, dim=0)  # (Nf, 4)
        meta.objects_n = torch.tensor(objects_n, dtype=torch.float)  # (Nf,)
        meta.objects_f = torch.tensor(objects_f, dtype=torch.float)  # (Nf,)
        # Overwrite background bounding box to the default large one
        meta.bounds = self.bounds

        # Actually store updated items
        output.update(meta)
        output.meta.update(meta)
        return output

    @property
    def n_views(self): return len(self.cameras)

    @property
    def n_latents(self): return len(next(iter(self.cameras.values())))  # short for timestamp

    @property
    def frame_min(self): return self.frame_sample[0] if len(self.frame_sample) == 3 else min(self.frame_sample)

    @property
    def frame_max(self):
        middle = (self.frame_sample[1] if self.frame_sample[1] else self.n_frames_total) - 1  # None -> all frames are loaded
        return middle if len(self.frame_sample) == 3 else max(self.frame_sample)

    @property
    def frame_int(self): return self.frame_sample[2] if len(self.frame_sample) == 3 else -1  # error out if you call this when giving specific frames

    @property
    def frame_range(self):
        return np.clip(self.frame_max - self.frame_min, 1, None)

    @property
    def view_min(self): return self.view_sample[0] if len(self.view_sample) == 3 else min(self.view_sample)

    @property
    def view_max(self):
        middle = (self.view_sample[1] if self.view_sample[1] else self.n_view_total) - 1  # None -> all frames are loaded
        return middle if len(self.view_sample) == 3 else max(self.view_sample)

    @property
    def view_int(self): return self.view_sample[2] if len(self.view_sample) == 3 else -1  # error out if you call this when giving specific frames

    @property
    def view_range(self):
        return np.clip(self.view_max - self.view_min, 1, None)

    def t_to_frame(self, t):
        return int(t * (self.frame_max - self.frame_min) + self.frame_min + 1e-5)  # avoid out of bounds

    def frame_to_t(self, frame_index):
        return (frame_index - self.frame_min) / self.frame_range  # avoid division by 0

    def frame_to_latent(self, frame_index):
        return int((frame_index - self.frame_min) / self.frame_int + 1e-5)  # avoid out of bounds

    def camera_to_v(self, camera_index):
        return (camera_index - self.view_min) / self.view_range  # avoid division by 0

    def v_to_camera(self, v):
        return int(v * (self.view_max - self.view_min) + self.view_min + 1e-5)  # avoid out of bounds

    def camera_to_view(self, camera_index):
        return int((camera_index - self.view_min) / self.view_int + 1e-5)  # avoid out of bounds

    def get_metadata(self, index):
        view_index, latent_index, camera_index, frame_index = self.get_indices(index)
        w2c, c2w, R, T, H, W, K, n, f, t = self.get_camera_params(view_index, latent_index)

        # NOTE: everything meta in the dataset are ignored when copying to cuda (avoiding syncing)
        meta = dotdict()  # camera parameters
        meta.H, meta.W = H, W
        meta.K, meta.R, meta.T = K, R, T
        meta.n, meta.f = n, f
        meta.w2c, meta.c2w = w2c, c2w
        meta.view_index, meta.latent_index, meta.camera_index, meta.frame_index = view_index, latent_index, camera_index, frame_index
        meta.t = t if self.use_loaded_time else self.frame_to_t(frame_index)
        meta.t = torch.as_tensor(meta.t, dtype=torch.float)  # the dataset provided time or the time fraction
        meta.v = self.camera_to_v(camera_index)
        meta.v = torch.as_tensor(meta.v, dtype=torch.float)  # the time fraction
        meta.n_rays = self.n_rays
        if self.use_aligned_cameras: meta.c2w_avg = self.c2w_avg  # MARK: store the aligned cameras here

        # Other inputs
        meta.bounds = self.get_bounds(latent_index)

        if self.use_smpls:
            smpl_poses, smpl_shapes, smpl_Rh, smpl_Th = self.get_smpl_motions(latent_index)
            meta.smpl_motions = dotdict()
            meta.smpl_motions.poses = smpl_poses
            meta.smpl_motions.shapes = smpl_shapes
            meta.smpl_motions.Rh = smpl_Rh
            meta.smpl_motions.Th = smpl_Th

        output = dotdict()
        output.update(meta)  # will also store a copy of these metadata on GPU
        output.meta = dotdict()  # this is the first time that this metadata is created in the batch
        output.meta.update(meta)

        # Maybe crop intrinsics
        if self.imbound_crop:
            self.crop_ixts_bounds(output)  # only crop target ixts

        # Maybe load foreground object priors
        if self.use_objects_priors:
            self.get_objects_priors(output)

        return output

    @staticmethod
    def scale_ixts(output: dotdict, ratio: float):
        orig_h, orig_w = output.H, output.W
        new_h, new_w = int(orig_h * ratio), int(orig_w * ratio)
        ratio_h, ratio_w = new_h / orig_h, new_w / orig_w
        K = output.K.clone()
        K[0:1] *= ratio_w
        K[1:2] *= ratio_h
        meta = dotdict()
        meta.K = K
        meta.tar_ixt = K
        meta.H = torch.as_tensor(new_h)
        meta.W = torch.as_tensor(new_w)
        if 'orig_h' in output:
            meta.crop_x = torch.as_tensor(int(output.crop_x * ratio))
            meta.crop_y = torch.as_tensor(int(output.crop_y * ratio))  # TODO: this is messy
            meta.orig_h = torch.as_tensor(int(output.orig_h * ratio))
            meta.orig_w = torch.as_tensor(int(output.orig_w * ratio))
            # Now the K corresponds to crop_x * self.render_ratio instead of int(output.crop_x * self.render_ratio)
            # We should fix that: no resizing, only a fractional movement here
            meta.K[..., :2, -1] -= torch.as_tensor([output.crop_x * ratio - meta.crop_x,  # only dealing with the cropped fraction
                                                    output.crop_y * ratio - meta.crop_y,  # only dealing with the cropped fraction
                                                    ],
                                                   device=output.bounds.device)  # crop K
        output.update(meta)
        output.meta.update(meta)
        return output

    @staticmethod
    def crop_ixts(output: dotdict, x, y, w, h):
        """
        Crops target intrinsics using a xywh 
        """
        K = output.K.clone()
        K[..., :2, -1] -= torch.as_tensor([x, y], device=output.bounds.device)  # crop K

        output.K = K
        output.tar_ixt = K

        meta = dotdict()
        meta.K = K
        meta.H = torch.as_tensor(h, device=output.bounds.device)
        meta.W = torch.as_tensor(w, device=output.bounds.device)
        if 'crop_x' in output.meta:
            meta.crop_x = torch.as_tensor(x + output.meta.crop_x, device=output.bounds.device)
            meta.crop_y = torch.as_tensor(y + output.meta.crop_y, device=output.bounds.device)
        else:
            meta.crop_x = torch.as_tensor(x, device=output.bounds.device)
            meta.crop_y = torch.as_tensor(y, device=output.bounds.device)
            meta.orig_w = output.W  # original size before update
            meta.orig_h = output.H  # original size before update
        output.update(meta)
        output.meta.update(meta)

        return output

    def get_ground_truth(self, index):
        # Load actual images, mask, sampling weights
        output = self.get_metadata(index)
        rgb, msk, wet, dpt, bkg = self.get_image(output.view_index, output.latent_index)  # H, W, 3
        H, W = rgb.shape[:2]
        output.rgb = rgb.view(-1, 3)  # full image in case you need it
        output.msk = msk.view(-1, 1)  # full mask (weights)
        output.wet = wet.view(-1, 1)  # full mask (weights)
        if dpt is not None: output.dpt = dpt.view(-1, 1)  # full depth image
        if bkg is not None: output.bkg = bkg.view(-1, 3)  # full background image

        # Maybe crop images
        if self.imbound_crop:  # crop_x has already been set by imbound_crop for ixts
            output = self.crop_imgs_bounds(output)  # only crop target imgs
            H, W = output.H.item(), output.W.item()
        elif self.immask_crop:  # these variables are only available when loading gts
            meta = dotdict()
            meta.crop_x = self.crop_xs[output.view_index, output.latent_index]
            meta.crop_y = self.crop_ys[output.view_index, output.latent_index]
            meta.orig_h = self.orig_hs[output.view_index, output.latent_index]
            meta.orig_w = self.orig_ws[output.view_index, output.latent_index]
            output.update(meta)
            output.meta.update(meta)

        # FIXME: Should add mutex to protect this， for now, multi-process and dataloading doesn't work well with each other
        # If Moderators are used, should set num_workers to 0 for single-process data loading
        n_rays = self.n_rays
        patch_size = self.patch_size
        render_ratio = self.render_ratio
        random_crop_size = self.random_crop_size
        render_center_crop_ratio = self.render_center_crop_ratio

        # Prepare for a different rendering ratio
        if (len(render_ratio.shape) and  # avoid length of 0-d tensor error, check length of shape
                render_ratio[output.view_index] != 1.0) or \
                render_ratio != 1.0:
            render_ratio = self.render_ratio[output.view_index] if len(self.render_ratio.shape) else self.render_ratio
            H, W = output.H.item(), output.W.item()
            rgb = output.rgb.view(H, W, 3)
            msk = output.msk.view(H, W, 1)
            wet = output.wet.view(H, W, 1)
            if dpt is not None: dpt = output.dpt.view(H, W, 1)
            if bkg is not None: bkg = output.bkg.view(H, W, 3)

            output = self.scale_ixts(output, render_ratio)
            H, W = output.H.item(), output.W.item()

            rgb = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(rgb)
            msk = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(msk)
            wet = as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(wet)
            if dpt is not None: as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(dpt)
            if bkg is not None: as_torch_func(partial(cv2.resize, dsize=(W, H), interpolation=cv2.INTER_AREA))(bkg)

            output.rgb = rgb.reshape(-1, 3)  # full image in case you need it
            output.msk = msk.reshape(-1, 1)  # full mask (weights)
            output.wet = wet.reshape(-1, 1)  # full mask (weights)
            if dpt is not None: output.dpt = dpt.reshape(-1, 1)
            if bkg is not None: output.bkg = bkg.reshape(-1, 1)

        # Prepare for a different rendering center crop ratio
        if (len(render_center_crop_ratio.shape) and  # avoid length of 0-d tensor error, check length of shape
                render_center_crop_ratio[output.view_index] != 1.0) or \
                render_center_crop_ratio != 1.0:
            render_center_crop_ratio = self.render_center_crop_ratio[output.view_index] if len(self.render_center_crop_ratio.shape) else self.render_center_crop_ratio
            H, W = output.H.item(), output.W.item()
            rgb = output.rgb.view(H, W, 3)
            msk = output.msk.view(H, W, 1)
            wet = output.wet.view(H, W, 1)
            if dpt is not None: dpt = output.dpt.view(H, W, 1)
            if bkg is not None: bkg = output.bkg.view(H, W, 3)

            w, h = int(W * render_center_crop_ratio), int(H * render_center_crop_ratio)
            x, y = w // 2, h // 2

            # Center crop the target image
            rgb = rgb[y: y + h, x: x + w, :]
            msk = msk[y: y + h, x: x + w, :]
            wet = wet[y: y + h, x: x + w, :]
            if dpt is not None: dpt[y: y + h, x: x + w, :]
            if bkg is not None: bkg[y: y + h, x: x + w, :]

            output.rgb = rgb.reshape(-1, 3)  # full image in case you need it
            output.msk = msk.reshape(-1, 1)  # full mask
            output.wet = wet.reshape(-1, 1)  # full weights
            if dpt is not None: output.dpt = dpt.reshape(-1, 1)
            if bkg is not None: output.bkg = bkg.reshape(-1, 1)

            # Crop the intrinsics
            self.crop_ixts(output, x, y, w, h)

        should_sample_patch = False
        should_crop_ixt = False

        # Source images will not walk in this path, thus we crop the ixts and images here
        if random_crop_size[0] > 0 and self.split == DataSplit.TRAIN:
            assert len(random_crop_size) == 2, 'Patch size should be a tuple of 2: height, width'
            Hp, Wp = random_crop_size
            Hp, Wp = Hp.long().item(), Wp.long().item()
            should_sample_patch = True
            should_crop_ixt = True

        # Sample random patch on images, but do not modify camera parameters
        if patch_size[0] > 0 and self.split == DataSplit.TRAIN:
            assert len(patch_size) == 2, 'Patch size should be a tuple of 2: height, width'
            Hp, Wp = patch_size
            Hp, Wp = Hp.long().item(), Wp.long().item()
            should_sample_patch = True
            should_crop_ixt = False

        if should_sample_patch:
            assert n_rays == -1, 'When performing patch sampling, do not resample rays on it'
            # Prepare images for patch sampling
            rgb = output.rgb.view(H, W, 3)
            msk = output.msk.view(H, W, 1)
            wet = output.wet.view(H, W, 1)
            if dpt is not None: dpt = output.dpt.view(H, W, 1)
            if bkg is not None: bkg = output.bkg.view(H, W, 3)

            # Find the Xp Yp Wp Hp to be used for random patch sampling
            # x = 0 if W - Wp <= 0 else np.random.randint(0, W - Wp + 1)
            # y = 0 if H - Hp <= 0 else np.random.randint(0, H - Hp + 1)
            # w = min(W, Wp)
            # h = min(H, Hp)
            at_least = min(Hp, Wp) // 4
            x = np.random.randint(-Wp + 1 + at_least, W - at_least)  # left edge
            y = np.random.randint(-Hp + 1 + at_least, H - at_least)  # left edge
            r = min(W, Wp + x)
            d = min(H, Hp + y)
            x = max(0, x)
            y = max(0, y)
            w = r - x
            h = d - y

            # Sample patches from the images
            rgb = rgb[y: y + h, x: x + w, :]
            msk = msk[y: y + h, x: x + w, :]
            wet = wet[y: y + h, x: x + w, :]
            if dpt is not None: dpt = dpt[y: y + h, x: x + w, :]
            if bkg is not None: bkg = bkg[y: y + h, x: x + w, :]

            output.rgb = rgb.reshape(-1, 3)  # full image in case you need it
            output.msk = msk.reshape(-1, 1)  # full mask
            output.wet = wet.reshape(-1, 1)  # full weights
            if dpt is not None: output.dpt = dpt.reshape(-1, 1)
            if bkg is not None: output.bkg = bkg.reshape(-1, 1)

        if should_crop_ixt:
            # Prepare the resized ixts
            self.crop_ixts(output, x, y, w, h)
        elif should_sample_patch:
            # Prepare the full sampling output
            H, W = output.H, output.W
            K, R, T = output.K, output.R, output.T

            # Calculate the pixel coordinates
            ray_o, ray_d, coords = get_rays(H, W, K, R, T, z_depth=self.use_z_depth, correct_pix=self.correct_pix, ret_coord=True)  # maybe without normalization
            ray_o = ray_o[y: y + h, x: x + w, :]
            ray_d = ray_d[y: y + h, x: x + w, :]
            coords = coords[y: y + h, x: x + w, :]

            # Prepare for computing loss on patch
            meta = dotdict()
            meta.patch_h = torch.as_tensor(h)
            meta.patch_w = torch.as_tensor(w)
            output.update(meta)
            output.meta.update(meta)

            # Store full sampling output
            output.ray_o = ray_o.reshape(-1, 3)  # full coords
            output.ray_d = ray_d.reshape(-1, 3)  # full coords
            output.coords = coords.reshape(-1, 2)  # full coords

        return output

    def __getitem__(self, index: int):  # for now, we are using the dict as list
        # Prepare the local timer in this multi-process dataloading environment
        local_timer = Timer(disabled=timer.disabled, sync_cuda=timer.sync_cuda)  # use global timer as reference
        local_timer.record('post processing')

        # Load ground truth
        output = self.get_ground_truth(index)  # load images, camera parameters, etc (10ms)
        local_timer.record('get ground truth')
        if 'ray_o' in output or self.n_rays < 0: return output  # directly return for the whole image (same for train and test)

        # Prepare weights for sampling
        H, W = output.H, output.W
        K, R, T = output.K, output.R, output.T
        rgb = output.rgb.view(H, W, 3)
        msk = output.msk.view(H, W, 1)
        wet = output.wet.view(H, W, 1)
        if 'dpt' in output: dpt = output.dpt.view(H, W, 1)
        if 'bkg' in output: bkg = output.bkg.view(H, W, 3)

        # Sample rays
        ray_o, ray_d, coords = weighted_sample_rays(wet,
                                                    K, R, T,
                                                    self.n_rays if self.split == DataSplit.TRAIN else -1,
                                                    self.use_z_depth,
                                                    self.correct_pix)  # N, 3; N, 3; N, 3; N, 2 (100ms)

        # Access and fetch data
        i, j = coords.unbind(-1)
        rgb = rgb[i, j]
        msk = msk[i, j]
        wet = wet[i, j]
        if 'dpt' in output: dpt = dpt[i, j]
        if 'bkg' in output: bkg = bkg[i, j]
        local_timer.record('weighted sample rays')

        # Main inputs
        output.rgb = rgb  # ground truth
        output.msk = msk
        output.wet = wet
        if 'dpt' in output: output.dpt = dpt
        if 'bkg' in output: output.bkg = bkg
        output.ray_o = ray_o
        output.ray_d = ray_d
        output.coords = coords
        return output

    def __len__(self): return self.n_views * self.n_latents  # there's no notion of epoch here

    @staticmethod
    def crop_ixts_bounds(output: dotdict):
        """
        Crops target intrinsics using a xywh computed from a bounds
        """
        x, y, w, h = get_bound_2d_bound(output.bounds, output.K, output.R, output.T, output.meta.H, output.meta.W)
        return VolumetricVideoDataset.crop_ixts(output, x, y, w, h)

    @staticmethod
    def crop_imgs_bounds(output: dotdict):
        """
        Crops target images using a xywh computed from a bounds and stored in the metadata
        """
        x, y, w, h = output.crop_x, output.crop_y, output.W, output.H
        H, W = output.orig_h, output.orig_w

        output.rgb = output.rgb.view(H, W, -1)[y:y + h, x:x + w].reshape(-1, 3)
        output.msk = output.msk.view(H, W, -1)[y:y + h, x:x + w].reshape(-1, 1)
        output.wet = output.wet.view(H, W, -1)[y:y + h, x:x + w].reshape(-1, 1)
        return output

    def get_viewer_batch(self, output: dotdict):
        # Source indices
        t = output.t
        v = output.v
        bounds = output.bounds  # camera bounds
        frame_index = self.t_to_frame(t)
        camera_index = self.v_to_camera(v)
        latent_index = self.frame_to_latent(frame_index)
        view_index = self.camera_to_view(camera_index)

        # Update indices, maybe not needed
        output.view_index = view_index
        output.frame_index = frame_index
        output.camera_index = camera_index
        output.latent_index = latent_index
        output.meta.view_index = view_index
        output.meta.frame_index = frame_index
        output.meta.camera_index = camera_index
        output.meta.latent_index = latent_index

        output.bounds = self.get_bounds(latent_index)  # will crop according to batch bounds
        output.bounds[0] = torch.maximum(output.bounds[0], bounds[0])  # crop according to user bound
        output.bounds[1] = torch.minimum(output.bounds[1], bounds[1])
        output.meta.bounds = output.bounds

        output = self.scale_ixts(output, self.render_ratio)

        if self.imbound_crop:
            output = self.crop_ixts_bounds(output)

        if self.use_objects_priors:
            output = self.get_objects_priors(output)

        return output  # how about just passing through
