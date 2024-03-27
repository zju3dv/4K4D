"""
Realtime4DV with background
Performs simple composition rendering with alpha blending and not fancy resizing
Potentially optimizable up to the rendering speed difference of high-res and low-res ones
"""

import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.engine import SAMPLERS, EMBEDDERS, REGRESSORS
from easyvolcap.models.samplers.uniform_sampler import UniformSampler
from easyvolcap.models.samplers.r4dv_sampler import R4DVSampler
from easyvolcap.models.networks.noop_network import NoopNetwork
from easyvolcap.utils.net_utils import load_network, freeze_module, VolumetricVideoModule
from easyvolcap.utils.image_utils import crop_using_mask
from easyvolcap.utils.ibr_utils import compute_src_inps
from easyvolcap.utils.timer_utils import timer


@SAMPLERS.register_module()
class R4DVBSampler(R4DVSampler):
    def __init__(self,
                 network: NoopNetwork,
                 fg_sampler_cfg: dotdict = dotdict(type=R4DVSampler.__name__),
                 bg_sampler_cfg: dotdict = dotdict(type=R4DVSampler.__name__),

                 pretrained_fg_sampler: str = '',
                 pretrained_bg_sampler: str = '',

                 freeze_fg_pcd: bool = False,
                 freeze_bg_pcd: bool = False,

                 freeze_fg_resd: bool = False,
                 freeze_bg_resd: bool = False,

                 freeze_fg_geo: bool = False,
                 freeze_bg_geo: bool = False,

                 **kwargs,
                 ):
        VolumetricVideoModule.__init__(self, network, **kwargs)
        self.fg_sampler: R4DVSampler = SAMPLERS.build(fg_sampler_cfg, network=network, **kwargs)
        self.bg_sampler: R4DVSampler = SAMPLERS.build(bg_sampler_cfg, network=network, n_frames=1, frame_sample=[0, 1, 1], **kwargs)

        load_network(self.fg_sampler, pretrained_fg_sampler, prefix='sampler.')  # hope these will work
        load_network(self.bg_sampler, pretrained_bg_sampler, prefix='sampler.')  # hope these will work

        # fmt: off
        if freeze_fg_pcd or freeze_fg_geo or freeze_fg_resd:
            if hasattr(self.fg_sampler, 'pcds'):           freeze_module(self.fg_sampler.pcds)
        if freeze_bg_pcd or freeze_bg_geo or freeze_bg_resd:
            if hasattr(self.bg_sampler, 'pcds'):           freeze_module(self.bg_sampler.pcds)
        if freeze_fg_pcd or freeze_fg_resd:
            if hasattr(self.fg_sampler, 'pcd_embedder'):   freeze_module(self.fg_sampler.pcd_embedder)
            if hasattr(self.fg_sampler, 'resd_regressor'): freeze_module(self.fg_sampler.resd_regressor)
        if freeze_bg_pcd or freeze_bg_resd:
            if hasattr(self.bg_sampler, 'pcd_embedder'):   freeze_module(self.bg_sampler.pcd_embedder)
            if hasattr(self.bg_sampler, 'resd_regressor'): freeze_module(self.bg_sampler.resd_regressor)
        if freeze_fg_geo:
            if hasattr(self.fg_sampler, 'xyz_embedder'):   freeze_module(self.fg_sampler.xyz_embedder)
            if hasattr(self.fg_sampler, 'geo_regressor'):  freeze_module(self.fg_sampler.geo_regressor)
        if freeze_bg_geo:
            if hasattr(self.bg_sampler, 'xyz_embedder'):   freeze_module(self.bg_sampler.xyz_embedder)
            if hasattr(self.bg_sampler, 'geo_regressor'):  freeze_module(self.bg_sampler.geo_regressor)
        # fmt: on

    def forward(self, batch: dotdict):
        # Global preparations (decoding)
        compute_src_inps(batch, 'src_inps').to(self.fg_sampler.dtype)  # will get decoded image and update batch
        compute_src_inps(batch, 'src_msks').to(self.fg_sampler.dtype)

        # Forward the foreground layer
        # Preparing source image
        store = dotdict()
        store.src_inps = batch.src_inps
        store.src_msks = batch.src_msks
        store.src_ixts = batch.src_ixts
        batch.src_ixts, batch.src_inps, batch.src_msks = crop_using_mask(batch.src_msks.permute(0, 1, 3, 4, 2),
                                                                         batch.src_ixts,
                                                                         batch.src_inps.permute(0, 1, 3, 4, 2),
                                                                         batch.src_msks.permute(0, 1, 3, 4, 2),
                                                                         )
        batch.src_inps = batch.src_inps.permute(0, 1, 4, 2, 3)
        batch.src_msks = batch.src_msks.permute(0, 1, 4, 2, 3)
        batch.src_inps = batch.src_inps * batch.src_msks

        fg_pcd, fg_xyz, fg_rgb, fg_rad, fg_occ = self.fg_sampler.forward(batch=batch, return_frags=True)

        # House keeping
        # Restoring source image
        batch.src_inps = store.src_inps
        batch.src_msks = store.src_msks
        batch.src_ixts = store.src_ixts

        # Forward the background layer
        store.latent_index = batch.latent_index
        store.meta.latent_index = batch.meta.latent_index
        batch.latent_index = torch.zeros_like(batch.latent_index)
        batch.meta.latent_index = torch.zeros_like(batch.meta.latent_index)
        # batch.src_inps = batch.src_inps * (1 - batch.src_msks)
        bg_pcd, bg_xyz, bg_rgb, bg_rad, bg_occ = self.bg_sampler.forward(batch=batch, return_frags=True)

        pcd, xyz, rgb, rad, occ = [
            torch.cat([fg_pcd, bg_pcd], dim=-2),
            torch.cat([fg_xyz, bg_xyz], dim=-2),
            torch.cat([fg_rgb, bg_rgb], dim=-2),
            torch.cat([fg_rad, bg_rad], dim=-2),
            torch.cat([fg_occ, bg_occ], dim=-2)]

        # Perform points rendering
        rgb, acc, dpt = self.bg_sampler.render_points(xyz, rgb, rad, occ, batch)  # B, HW, C
        self.bg_sampler.store_output(pcd, xyz, rgb, acc, dpt, batch)

        # House keeping
        # batch.src_inps = batch.old_src_inps
        batch.latent_index = store.latent_index
        batch.meta.latent_index = store.meta.latent_index
        for k in list(batch.keys()):
            if k.startswith('old'):
                del batch[k]
        for k in list(batch.meta.keys()):
            if k.startswith('old'):
                del batch.meta[k]

        if 'msk' in batch:
            batch.msk = torch.ones_like(batch.msk)  # gatekeeping for training
        timer.record('rendering')
