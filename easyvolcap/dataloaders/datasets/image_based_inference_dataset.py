import torch
from typing import List
from easyvolcap.engine import DATASETS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import *
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.dataloaders.datasets.volumetric_video_inference_dataset import VolumetricVideoInferenceDataset


@DATASETS.register_module()
class ImageBasedInferenceDataset(VolumetricVideoInferenceDataset):
    def __init__(self,
                 n_srcs_list: List[int] = [3],  # MARK: repeated global configuration
                 n_srcs_prob: List[int] = [1.0],  # MARK: repeated global configuration
                 append_gt_prob: float = 1.0,
                 extra_src_pool: int = 1,
                 supply_decoded: bool = False,
                 barebone: bool = False,

                 #  closest_using_t: bool = False,
                 #  src_view_sample: List[int] = [0, None, 1],  # use these as input source views

                 **kwargs,
                 ):
        # NOTE: This file inherits from VolumetricVideoInferenceDataset instead of the ImageBasedDataset
        # Thus functions reusing implementation from that class should explicit define this
        # self.closest_using_t = closest_using_t  # MARK: transpose
        # self.src_view_sample = src_view_sample
        call_from_cfg(super().__init__, kwargs)  # will have prepared other parts of the dataset (interpolation or orbit)
        if self.src_view_sample != [0, None, 1] and self.view_sample != [0, None, 1]: log(yellow(f'Using `src_view_sample = {self.src_view_sample}` when `view_sample = {self.view_sample}` is not default'))

        # ImageBasedDataset.load_source_params(self)  # no extra dependencies
        ImageBasedDataset.load_source_indices(self)  # no extra dependencies
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob
        self.extra_src_pool = extra_src_pool
        self.append_gt_prob = append_gt_prob  # manually assign values

        # src_inps will come in as decoded bytes instead of jpegs
        self.supply_decoded = supply_decoded
        self.barebone = barebone

    def load_interpolations(self):
        ImageBasedDataset.load_source_params(self)  # remember things

        # Actual interpolation
        super().load_interpolations()

        # For physical to virtual indexing (ibr inference uses physical indexing)
        # While the original inference dataset uses virtual indexing
        # fmt: off
        self.Hs    = self.Hs  .expand(-1, self.src_ixts.shape[1], *self.Hs.shape[2:]  )
        self.Ws    = self.Ws  .expand(-1, self.src_ixts.shape[1], *self.Ws.shape[2:]  )
        self.Ks    = self.Ks  .expand(-1, self.src_ixts.shape[1], *self.Ks.shape[2:]  )
        self.Rs    = self.Rs  .expand(-1, self.src_ixts.shape[1], *self.Rs.shape[2:]  )
        self.Ts    = self.Ts  .expand(-1, self.src_ixts.shape[1], *self.Ts.shape[2:]  )
        self.Cs    = self.Cs  .expand(-1, self.src_ixts.shape[1], *self.Cs.shape[2:]  )
        self.c2ws  = self.c2ws.expand(-1, self.src_ixts.shape[1], *self.c2ws.shape[2:])
        self.w2cs  = self.w2cs.expand(-1, self.src_ixts.shape[1], *self.w2cs.shape[2:])
        # fmt: on

    def load_paths(self):
        return VolumetricVideoDataset.load_paths(self)  # store images names

    def load_bytes(self):
        return VolumetricVideoDataset.load_bytes(self)  # store images

    def virtual_to_physical(self, latent_index: int):
        return VolumetricVideoDataset.virtual_to_physical(self, latent_index)

    def physical_to_virtual(self, latent_index: int):
        return VolumetricVideoDataset.physical_to_virtual(self, latent_index)

    def get_objects_bounds(self, latent_index: int):
        return VolumetricVideoDataset.get_objects_bounds(self, latent_index)

    def get_objects_priors(self, output: dotdict):
        return VolumetricVideoDataset.get_objects_priors(self, output)

    def load_source_params(self):
        return ImageBasedDataset.load_source_params(self)

    def __getitem__(self, index: dotdict):
        return ImageBasedDataset.get_metadata(self, index)

    # Manual polymorphism from ImageBasedDataset
    # NOTE: Refactor will never respect this
    @staticmethod
    def crop_ixts_xywh(size: List[int], output: dotdict):
        return ImageBasedDataset.crop_ixts_xywh(size, output)

    @staticmethod
    def crop_imgs_xywh(size: List[int], output: dotdict):
        return ImageBasedDataset.crop_imgs_xywh(size, output)

    @staticmethod
    def crop_srcs_mask(output: dotdict):
        return ImageBasedDataset.crop_srcs_mask(output)

    @staticmethod
    def crop_tars_mask(output: dotdict):
        return ImageBasedDataset.crop_tars_mask(output)

    def get_viewer_batch(self, batch):
        return ImageBasedDataset.get_viewer_batch(self, batch)

    def get_sources(self, *args, **kwargs):
        return ImageBasedDataset.get_sources(self, *args, **kwargs)
