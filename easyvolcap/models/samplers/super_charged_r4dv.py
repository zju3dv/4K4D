# This is a inference sampler for the turbo-charged point-planes model.
from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Tuple
if TYPE_CHECKING:
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
    from easyvolcap.runners.volumetric_video_viewer import VolumetricVideoViewer

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from types import MethodType

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS, EMBEDDERS, REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.timer_utils import timer
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.bound_utils import get_bounds
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.math_utils import affine_inverse, normalize
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.data_utils import to_cuda, to_tensor, add_batch
from easyvolcap.utils.enerf_utils import sample_geometry_feature_image
from easyvolcap.utils.ibr_utils import compute_src_feats, compute_src_inps
from easyvolcap.utils.cuda_utils import register_memory, unregister_memory
from easyvolcap.utils.image_utils import interpolate_image, pad_image
from easyvolcap.utils.net_utils import unfreeze_module, freeze_module, typed, make_params, make_buffer

from easyvolcap.models.networks.embedders.kplanes_embedder import KPlanesEmbedder
from easyvolcap.models.samplers.point_planes_sampler import PointPlanesSampler
from easyvolcap.models.networks.embedders.geometry_image_based_embedder import GeometryImageBasedEmbedder
from easyvolcap.models.networks.regressors.image_based_spherical_harmonics import ImageBasedSphericalHarmonics

# This function will cache all point features by directly querying the xyz_embedder
def load_state_dict_kwargs(device: str = 'cuda'):
    runner: 'VolumetricVideoRunner' = cfg.runner  # assume the runner has all we need now
    dataset: 'ImageBasedDataset' = runner.val_dataloader.dataset
    n_views = dataset.n_latents if dataset.closest_using_t else dataset.n_views
    n_latents = dataset.n_views if dataset.closest_using_t else dataset.n_latents
    vb, ve, vs = dataset.view_sample if dataset.closest_using_t else dataset.frame_sample  # validatiaon
    b, e, s = vb, ve, vs  # use validation samples for computing indices

    src_inds = torch.arange(n_views, device=device)  # the original view sample index w.r.t trainingsrc_view_sample
    if hasattr(dataset, 'src_view_sample'):
        src_view_sample = dataset.src_view_sample
        if len(src_view_sample) != 3: src_inds = src_inds[src_view_sample]  # this is a list of indices
        else: src_inds = src_inds[src_view_sample[0]:src_view_sample[1]:src_view_sample[2]]  # begin, start, end
    n_srcs = len(src_inds)

    tb, te, ts = SuperChargedR4DV.frame_sample  # training
    times = torch.arange(tb, te, ts, device=device)
    times = (times - tb) / np.clip(te - 1 - tb, 1, None)  # MARK: same as frame_to_t in VolumetricVideoDataset
    kwargs = dotdict(b=b, e=e, s=s,
                     tb=tb, te=te, ts=ts,
                     times=times,
                     n_latents=n_latents,
                     n_views=n_views,
                     n_srcs=n_srcs,
                     src_inds=src_inds)
    return kwargs


@torch.no_grad()
def forward_for_xyz_feat(i: int, sampler: SuperChargedR4DV, b, e, s, tb, te, ts, times, **kwargs):
    xyz: torch.Tensor = sampler.pcds[(b + i * s) // ts - tb][None]
    t = times[(b + i * s) // ts - tb]
    xyz_t = t[None, None].expand(*xyz.shape[:-1], 1)  # B, N, 1
    timer.record('post processing')
    xyz_feat: torch.Tensor = sampler.xyz_embedder(xyz, xyz_t)  # same time
    timer.record('sampling 4D feature')
    return xyz, xyz_feat  # this is without batch dimension


@torch.no_grad()
def forward_for_pcd_feat(i: int, sampler: SuperChargedR4DV, b, e, s, tb, te, ts, times, **kwargs):
    pcd: torch.Tensor = sampler.pcds[(b + i * s) // ts - tb][None]
    t = times[(b + i * s) // ts - tb]
    pcd_t = t[None, None].expand(*pcd.shape[:-1], 1)  # B, N, 1
    pcd_feat: torch.Tensor = sampler.pcd_embedder(pcd, pcd_t)  # same time
    return pcd, pcd_feat  # this is without batch dimension


@torch.no_grad()
def average_single_frame(i: int,
                         sampler: SuperChargedR4DV,
                         dataset: ImageBasedDataset,
                         runner: VolumetricVideoRunner,
                         forward_for_xyz_feat=forward_for_xyz_feat,
                         **kwargs):  # MARK: very memory intensive & time intensive
    """
    Precomputes the blending for a single frame from multiple views.

    This function performs several operations to process a single frame from a volumetric video dataset. 
    It first computes the geometry features and then loads the camera parameters. It then fetches the 
    source images and processes them to match the desired input size for the subsequent neural network stages. 
    The image features are computed and blended using the IBR (image-based rendering) networks, and blending 
    weights are calculated. This function is memory and time intensive due to the large amount of data 
    processing and neural network computations.

    If this process could be accelerated, we'll be able to play some really long volumetric videos.
    We aim to tone down this preprocessing to just 33ms, which should make the playback smooth enough.
    """
    timer.record('move to cpu')

    # Load data from dataset and the sampler itsampler
    kwargs = dotdict(kwargs)
    n_views = kwargs.n_views
    xyz, xyz_feat = forward_for_xyz_feat(i)  # repeated computation
    src_inds = kwargs.src_inds.cpu().numpy().tolist()
    timer.record('load geometry feature')

    # Load source camera parameters
    src_ixts = dataset.src_ixts[:, i].to(xyz)[None]
    src_exts = dataset.src_exts[:, i].to(xyz)[None]  # B, S, 3, 3

    # Prepare optimized source camera parameters
    meta = dotdict(src_inds=kwargs.src_inds[None], t_inds=torch.as_tensor(i)[None])
    batch = dotdict(src_exts=src_exts, meta=meta)
    batch.update(meta)
    batch = to_cuda(batch)
    batch = runner.model.camera.forward_srcs(batch)
    src_exts = batch.src_exts
    timer.record('load cameras')

    # Load source images from the dataset
    if dataset.closest_using_t: src_inps = list(zip(*parallel_execution([i] * kwargs.n_srcs, src_inds, action=dataset.get_image)))[0]  # S: H, W, 3 # MARK: SYNC
    else: src_inps = list(zip(*parallel_execution(src_inds, [i] * kwargs.n_srcs, action=dataset.get_image)))[0]  # S: H, W, 3 # MARK: SYNC

    # Move the source images to GPU and concatenate them (with black scaling)
    src_inps = [inp[None].permute(0, 3, 1, 2).to(xyz) for inp in src_inps]  # S: B, 3, H, W  # move to the same device
    src_inps = compute_src_inps(dotdict(src_inps=src_inps))  # B, S, 3, H, W, there exists some cropping and filling with black here

    # Compute rendering size
    img_pad = sampler.ibr_embedder.feat_reg.size_pad
    Hc, Wc = src_inps.shape[-2:]  # padded and cropped image size
    Hp, Wp = int(np.ceil(Hc / img_pad)) * img_pad, int(np.ceil(Wc / img_pad)) * img_pad  # Input and output should be same in size
    src_inps = pad_image(src_inps, size=(Hp, Wp))  # B, S, 3, H, W
    timer.record('load source images')

    # Pass through the IBR networks for blending weights
    src_feat = torch.stack([compute_src_feats(
        inp,
        sampler.ibr_embedder.feat_reg
    )[-1] for inp in src_inps[0]])[None]  # B, S, 3, H, W
    timer.record('image feature extraction')

    src_feat_inps = torch.cat([
        src_feat,
        src_inps,
    ], dim=-3)  # B, S, C, H, W

    # Compute projected color of every image, using the original size image
    ibrs_rgbs = torch.cat([sample_geometry_feature_image(
        xyz,
        src_feat_inps[:, i:i + 1],
        src_exts[:, i:i + 1],
        src_ixts[:, i:i + 1],
        src_inps.new_ones(2, 1),
    ) for i in range(src_feat_inps.shape[1])], dim=1)  # B, S, N, 3
    ibrs, rgbs = ibrs_rgbs[..., :-3], ibrs_rgbs[..., -3:]
    del src_feat, src_inps, src_feat_inps, ibrs_rgbs
    timer.record('sample image features')

    exp_xyz_feat = xyz_feat[..., None, :, :].expand(ibrs.shape[:-1] + (xyz_feat.shape[-1],))
    xyz_ibr_rgbs = torch.cat([exp_xyz_feat, ibrs, rgbs], dim=-1)  # B, S, N, 43
    del exp_xyz_feat, ibrs

    # Compute blending weights from the image features
    bws = torch.cat([sampler.ibr_regressor.rgb_mlp(xyz_ibr_rgbs[:, j:j + 1]) for j in range(xyz_ibr_rgbs.shape[-3])], dim=-3)  # B, S, N, 1
    timer.record('compute blending weights')

    # Reshape for returning
    return torch.cat([rgbs, bws], dim=-1), affine_inverse(src_exts)[..., :3, 3]  # B, S, N, 4 & B, S, 3


@SAMPLERS.register_module()
class SuperChargedR4DV(PointPlanesSampler):
    def __init__(self,
                 dtype: torch.dtype = torch.half,
                 use_cudagl: bool = True,
                 skip_loading_points: bool = True,

                 ibr_embedder_cfg: dotdict = dotdict(type=GeometryImageBasedEmbedder.__name__),  # easily returns nan
                 ibr_regressor_cfg: dotdict = dotdict(type=ImageBasedSphericalHarmonics.__name__),  # easily returns nan

                 # Visualization
                 skip_shs: bool = False,
                 skip_base: bool = False,

                 *args,
                 **kwargs,
                 ):
        # Initialize the base class (trainable module)
        kwargs = dotdict(kwargs)
        self.kwargs = kwargs
        super().__init__(*args, **kwargs, dtype=dtype, use_cudagl=use_cudagl, skip_loading_points=skip_loading_points)

        del self.dir_embedder  # no need for this
        del self.rgb_regressor
        self.ibr_embedder: GeometryImageBasedEmbedder = EMBEDDERS.build(ibr_embedder_cfg)  # forwarding the images
        self.ibr_regressor: ImageBasedSphericalHarmonics = REGRESSORS.build(ibr_regressor_cfg, in_dim=self.xyz_embedder.out_dim + 3, src_dim=self.ibr_embedder.src_dim)
        self.pre_handle = self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        self.super_charge(**kwargs)

        self.skip_shs = skip_shs
        self.skip_base = skip_base

    def render_imgui(self, viewer: 'VolumetricVideoViewer', batch: dotdict):
        from imgui_bundle import imgui_toggle
        toggle_ios_style = imgui_toggle.ios_style(size_scale=0.2)
        self.skip_shs = imgui_toggle.toggle('Skip SHs', self.skip_shs, config=toggle_ios_style)[1]
        self.skip_base = imgui_toggle.toggle('Skip base', self.skip_base, config=toggle_ios_style)[1]

    def super_charge(self,
                     n_srcs: int = 4,
                     n_shs: int = 3,  # (n + 1) ** 2
                     cache_size: int = 10,  # hold 5 frames in gpu memory, others in cpu memory
                     memory_dtype: torch.dtype = torch.half,  # packing
                     compute_dtype: torch.dtype = torch.float,
                     should_release_memory: bool = True,

                     # Retain pcd kplanes
                     retain_resd: bool = False,

                     **kwargs,
                     ):
        # We extract this function out to reuse its components
        self.n_shs = n_shs
        self.n_srcs = n_srcs
        self.cache_size = cache_size
        self.retain_resd = retain_resd
        self.should_release_memory = should_release_memory
        self.ibr_sh_deg = self.ibr_regressor.sh_deg
        self.ibr_sh_dim = self.ibr_regressor.sh_dim
        self.ibr_out_dim = self.ibr_regressor.out_dim
        self.ibr_resd_limit = self.ibr_regressor.resd_limit
        if not retain_resd:
            del self.pcd_embedder
            del self.resd_regressor

        # Prepare for relevant data types
        self.memory_dtype = getattr(torch, memory_dtype) if isinstance(memory_dtype, str) else memory_dtype
        self.compute_dtype = getattr(torch, compute_dtype) if isinstance(compute_dtype, str) else compute_dtype

        # Initial control of the loaded datatype
        self.type(self.dtype)
        self.post_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

        # Prepare for streaming things
        self.streams: List[torch.cuda.Stream] = [torch.cuda.Stream() for _ in self.pcds]  # data moving streams
        self.cache: List[torch.Tensor] = [None for _ in self.pcds]
        self.indices: List[int] = []

    def state_dict(self, destination, prefix, keep_vars, **kwargs):
        super().state_dict(destination, prefix, keep_vars, **kwargs)
        return self._save_state_dict_pre_hook(destination, prefix, keep_vars, **kwargs)

    def _save_state_dict_pre_hook(self, destination, prefix, keep_vars, **kwargs):
        if hasattr(self, 'shs'):
            for i in range(len(self.shs)):
                destination[f'{prefix}shs.{i}'] = self.shs[i]
        if hasattr(self, 'rgbws'):
            for i in range(len(self.rgbws)):
                destination[f'{prefix}rgbws.{i}'] = self.rgbws[i]
        return destination

    def release_memory(self):
        del self.xyz_embedder
        del self.geo_regressor
        del self.ibr_embedder
        del self.ibr_regressor
        if self.retain_resd:
            del self.pcd_embedder
            del self.resd_regressor

    def fetch(self, indices: List[int], cpu_stores: List[List[torch.Tensor]]):
        return [self.fetch_one(l, cpu_stores) for l in indices]

    def fetch_one(self, i: int, cpu_stores: List[List[torch.Tensor]]):
        for j in range(self.cache_size):
            if i + j < len(self.cache) and self.cache[i + j] is None and len(cpu_stores) and cpu_stores[0][i + j] is not None and i + j not in self.indices:
                # 1. Not out of bound
                # 2. The cache was not being copied
                # 3. The store is not empty
                # 4. The content in the store is not empty
                # 5. Current indices is not in the cache, append it
                if len(self.indices) == self.cache_size:
                    # If maximum number of cache is reached, pop the oldest one (mostly current frame)
                    left = self.indices.pop(0)
                    self.cache[left] = None  # clear cache
                self.indices.append(i + j)
                with torch.cuda.stream(self.streams[i + j]):
                    self.cache[i + j] = [cpu_store[i + j].to('cuda', non_blocking=True).view(self.dtype) for cpu_store in cpu_stores]
        torch.cuda.current_stream().wait_stream(self.streams[i])
        return self.cache[i]

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        super()._load_state_dict_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        # Historical reasons
        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.feat_agg' in key:
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.rgb_mlp.linears' in key:
                state_dict[key.replace(f'{prefix}ibr_regressor.rgb_mlp.linears', f'{prefix}ibr_regressor.rgb_mlp.mlp.linears')] = state_dict[key]
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}ibr_regressor.sh_mlp.linears' in key:
                state_dict[key.replace(f'{prefix}ibr_regressor.sh_mlp.linears', f'{prefix}ibr_regressor.sh_mlp.mlp.linears')] = state_dict[key]
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(f'{prefix}cents.'):  # historical reasons
                del state_dict[key]

        for key in keys:
            if key.startswith(f'{prefix}sh_mlp.'):  # historical reasons
                del state_dict[key]

        keys = list(state_dict.keys())
        for key in keys:
            if key.startswith(f'{prefix}rgbws.'):  # historical reasons
                del state_dict[key]

        if not self.retain_resd:
            keys = list(state_dict.keys())
            for key in keys:
                if key.startswith(f'{prefix}resd_regressor.'):  # historical reasons
                    del state_dict[key]

    @torch.no_grad()
    def _load_state_dict_post_hook(self, module, incompatible_keys):
        runner: 'VolumetricVideoRunner' = cfg.runner  # assume the runner has all we need now
        dataset: 'ImageBasedDataset' = runner.val_dataloader.dataset
        kwargs = load_state_dict_kwargs(self.pcds[0].device)
        times = kwargs.times
        n_views = kwargs.n_views
        n_latents = kwargs.n_latents
        b, e, s = kwargs.b, kwargs.e, kwargs.s
        tb, te, ts = kwargs.tb, kwargs.te, kwargs.ts

        self.type(self.compute_dtype)

        # 2 ** 18 * 64 * 4 * 2 = 2 ** 27 = 128MB * 150 = 50-60GB
        self.rgbws = [None for _ in self.pcds]  # HUGE
        self.cents = nn.ParameterList([None for _ in self.pcds])  # OK

        # Lower memory & vram usage
        old_pcds = self.pcds
        self.pcds = nn.ParameterList([None for _ in self.pcds])
        for i in range(n_latents): self.pcds[(b + i * s) // ts - tb] = old_pcds[(b + i * s) // ts - tb]

        # Load tighter bounds from the trained models
        dataset.vhull_bounds = [dataset.bounds for _ in range(len(self.pcds))]
        for i in range(len(self.pcds)):
            if self.pcds[i] is not None:
                dataset.vhull_bounds[i] = get_bounds(self.pcds[i][None], padding=0.01)[0].cpu()  # MARK: SYNC

        l_forward_for_pcd_feat = partial(forward_for_pcd_feat, sampler=self, **kwargs)
        l_forward_for_xyz_feat = partial(forward_for_xyz_feat, sampler=self, **kwargs)
        l_average_single_frame = partial(average_single_frame, sampler=self, dataset=dataset, runner=runner, forward_for_xyz_feat=l_forward_for_xyz_feat, **kwargs)
        if self.retain_resd:
            for i in tqdm(range(n_latents), desc='Computing xyz for caching'):
                pcd, feat = l_forward_for_pcd_feat(i)
                resd = self.resd_regressor(feat)
                xyz = pcd + resd
                self.pcds[(b + i * s) // ts - tb] = make_buffer(xyz[0])

        for i in tqdm(range(n_latents), desc=f'Caching rgbw and center'):
            rgbw, cent = l_average_single_frame(i)
            rgbw = rgbw.to(self.dtype).view(self.memory_dtype).detach().cpu(memory_format=torch.contiguous_format)  # MARK: SYNC
            torch.cuda.empty_cache()  # only out-of-frame cache cleaning works
            rgbw = register_memory(rgbw)
            self.rgbws[(b + i * s) // ts - tb] = make_buffer(rgbw[0])
            self.cents[(b + i * s) // ts - tb] = make_buffer(cent[0])

        feats = []
        for i in tqdm(range(n_latents), desc='Computing features for caching'):
            xyz, feat = l_forward_for_xyz_feat(i)
            feat = feat.detach()
            feats.append(feat[0])

        if hasattr(self, 'ibr_regressor') and hasattr(self.ibr_regressor, 'sh_mlp'):
            self.shs = [None for _ in self.pcds]  # BIG
            for i, feat in enumerate(tqdm(feats, desc='Caching spherical harmonics')):
                timer.record('sh overhead')
                sh: torch.Tensor = self.ibr_regressor.sh_mlp(feat)
                timer.record('sh regression')
                sh = sh.view(*sh.shape[:-1], self.ibr_out_dim, self.ibr_sh_dim // self.ibr_out_dim)[..., :(self.n_shs + 1) ** 2]  # reshape to B, P, 3, SH
                sh = sh.to(self.dtype).view(self.memory_dtype).detach().cpu(memory_format=torch.contiguous_format)  # MARK: SYNC
                torch.cuda.empty_cache()  # only out-of-frame cache cleaning works
                sh = register_memory(sh)  # only registering using cudart
                self.shs[(b + i * s) // ts - tb] = make_buffer(sh)

        self.precompute_geometry(feats, b, s, ts, tb)

        self.type(self.dtype)

        # Release memory of dataset
        if self.should_release_memory:
            dataset.ims_bytes = None
            dataset.bgs_bytes = None
            dataset.mks_bytes = None

            self.release_memory()

            # Give OpenGL some breathing room
            torch.cuda.empty_cache()

    def precompute_geometry(self, feats: List[torch.Tensor], b, s, ts, tb):
        self.rads = nn.ParameterList([None for _ in self.pcds])  # OK
        self.occs = nn.ParameterList([None for _ in self.pcds])  # OK
        for i, feat in enumerate(tqdm(feats, desc='Caching radius and alpha')):
            timer.record('geometry overhead')
            rad, occ = self.geo_regressor(feat)
            timer.record('geometry regression')
            self.rads[(b + i * s) // ts - tb] = make_buffer(rad)
            self.occs[(b + i * s) // ts - tb] = make_buffer(occ)

    @staticmethod
    @torch.jit.script
    def get_inds(cent: torch.Tensor, C: torch.Tensor, n_srcs: int = 4):
        diff = cent - C
        sims = 1 / (diff * diff).sum(dim=-1).clip(1e-10)  # B, S
        inds = torch.topk(sims, k=n_srcs, dim=-1, sorted=False, largest=True)[-1]  # B, 4
        return inds

    @staticmethod
    @torch.jit.script
    def get_base(rgbw: torch.Tensor):
        base = (rgbw[..., -1:].softmax(-3) * rgbw[..., :-1]).sum(-3)
        return base

    @staticmethod
    @torch.jit.script
    def get_rgbw(rgbw: torch.Tensor, inds: torch.Tensor):
        # rgbw = multi_gather(rgbw, inds, dim=-3)  # B, 4, N, 4
        inds = inds[..., None, None].expand((-1, -1) + rgbw.shape[-2:])  # B, 4, N, 4
        rgbw = rgbw.gather(-3, inds)  # B, 4, N, 4
        return rgbw

    @staticmethod
    @torch.jit.script
    def get_dir(xyz: torch.Tensor, C: torch.Tensor):
        dir = normalize(xyz.detach() - C)
        return dir

    @staticmethod
    @torch.jit.script
    def get_rgb(base: torch.Tensor, sh: torch.Tensor, dir: torch.Tensor, sh_deg: int = 3, resd_limit: float = 1.0):
        rgb = base + eval_sh(sh_deg, sh, dir).tanh() * resd_limit  # NOTE: this is the only thing that need to be run on CUDA (or torch)
        rgb = rgb.clip(0, 1)
        return rgb

    @staticmethod
    @torch.jit.script
    def get_rgb(R: torch.Tensor, T: torch.Tensor, xyz: torch.Tensor, sh: torch.Tensor, rgbw: torch.Tensor, cent: torch.Tensor, n_srcs: int = 4, sh_deg: int = 3, resd_limit: float = 1.0):
        # MARK: 0.8-0.9ms
        C = (-R.mT @ T).mT
        dir = normalize(xyz.detach() - C)

        # IBR
        diff = cent - C
        sims = 1 / (diff * diff).sum(dim=-1).clip(1e-10)  # B, S
        inds = torch.topk(sims, k=n_srcs, dim=-1, sorted=False, largest=True)[-1]  # B, 4
        inds = inds[..., None, None].expand((-1, -1) + rgbw.shape[-2:])  # B, 4, N, 4
        rgbw = rgbw.gather(-3, inds)  # B, 4, N, 4
        base = (rgbw[..., -1:].softmax(-3) * rgbw[..., :-1]).sum(-3)
        
        # Residual speculars
        rgb = base + eval_sh(sh_deg, sh, dir).tanh() * resd_limit  # NOTE: this is the only thing that need to be run on CUDA (or torch)
        rgb = rgb.clip(0, 1)
        return rgb

    def forward(self, batch: dotdict, return_frags: bool = False):
        # Get corresponding indices for sampling
        index, time = self.sample_index_time(batch)

        # Extract input
        xyz = torch.stack([self.pcds[l] for l in index])  # B, N, 3
        rad = torch.stack([self.rads[l] for l in index])  # B, N, 3
        occ = torch.stack([self.occs[l] for l in index])  # B, N, 3
        cent = torch.stack([self.cents[l] for l in index])  # B, S, 3

        # Compute rgb
        values = self.fetch(index, [self.shs, self.rgbws])  # will initiate copy for both rgbw and sh, trying to overlap them
        sh = torch.stack([v[0] for v in values])
        rgbw = torch.stack([v[1] for v in values])  # B, S, N, 4
        if self.skip_shs:
            sh[:] = 0
        if self.skip_base:
            sh = sh.abs()
            rgbw[..., :3] = 0
        timer.record('sample source images')

        rgb = self.get_rgb(batch.R.half(), batch.T.half(), xyz, sh, rgbw, cent, self.n_srcs, self.n_shs, self.ibr_resd_limit)
        timer.record('evaluate SH')

        if return_frags:
            return None, xyz, rgb, rad, occ

        # Perform points rendering (for now, this is dominating)
        rgb, acc, dpt = self.render_points(xyz, rgb, rad, occ, batch)  # almost always use render_cudagl
        timer.record('render points')

        # Prepare for output
        self.store_output(None, xyz, rgb, acc, dpt, batch)
        return None
