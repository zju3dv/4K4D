# This dataset creates instances of volumetric_video_dataset or image_based_dataset for all specified scenes
# And samples one of them during training of the generalizable model

import os
import copy
import torch
from glob import glob
from os.path import join
from typing import List, Union

from easyvolcap.engine import DATASETS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import DataSplit
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset


@DATASETS.register_module()
class GeneralizableDataset(VolumetricVideoDataset):
    def __init__(self,
                 # Consideration, should we expose all configurable entries through a multi-level network like design
                 # Or just leave everything as default?

                 # The final design:
                 # User could pass in addtional arguments through dataset configs
                 dataset_cfgs: List[dotdict] = [
                     dotdict(type=ImageBasedDataset.__name__),  # the last dataset here will be used as default during dataloading
                 ],

                 data_roots=[],  # this will overwrite the dataset configs
                 meta_roots=['data/dtu'],  # this will overwrite thet data_roots configs,

                 **kwargs,
                 ):

        # Prepare for dataset config
        if meta_roots and data_roots:
            log(yellow(f'data_roots entries will be replace by meta_roots: {meta_roots}'))

        if meta_roots:
            data_roots = []
            # Load dataset configs here
            # Will treat every subdirectory containing a images folder as a dataset folder
            for meta_root in meta_roots:
                meta_data_roots = sorted(glob(join(meta_root, '*')))
                for data_root in meta_data_roots:
                    if os.path.isdir(data_root):
                        if os.path.exists(join(data_root, 'images')):
                            data_roots.append(data_root)

        if data_roots and len(dataset_cfgs) > 1:
            log(yellow(f'dataset_cfgs entries will be replace by data_roots: {data_roots}'))

        if data_roots:
            # Load dataset configs here
            if len(dataset_cfgs):
                default_dataset_cfg = dataset_cfgs[-1]
            else:
                default_dataset_cfg = dotdict(type=ImageBasedDataset.__name__)

            dataset_cfgs = []
            for data_root in data_roots:
                dataset_cfg = default_dataset_cfg.copy()
                dataset_cfg.update(kwargs)
                dataset_cfg.data_root = data_root
                dataset_cfgs.append(dataset_cfg)

        # Reuse these reusable contents
        self.datasets: List[ImageBasedDataset] = [DATASETS.build(dataset_cfg) for dataset_cfg in dataset_cfgs]
        self.lengths = torch.as_tensor([len(d) for d in self.datasets])
        self.accum_lengths = self.lengths.cumsum(dim=-1)

    @property
    def render_ratio(self):
        return self.datasets[0].render_ratio_shared.item()

    @render_ratio.setter
    def render_ratio(self, value: float):
        for dataset in self.datasets:
            dataset.render_ratio_shared.fill_(value)

    def extract_dataset_index(self, index: Union[dotdict, int]):
        # Maybe think of a better way to update input of __getitem__
        if isinstance(index, dotdict): sampler_index, n_srcs = index.index, index.n_srcs
        else: sampler_index = index

        # Dataset index will indicate the sample to use
        dataset_index = torch.searchsorted(self.accum_lengths, sampler_index, right=True)  # 2 will not be inserted as 2
        sampler_index = sampler_index - (self.accum_lengths[dataset_index - 1] if dataset_index > 0 else 0)  # maybe -1
        # MARK: This is nasty, pytorch inconsistency of conversion
        sampler_index = sampler_index.item() if isinstance(sampler_index, torch.Tensor) else sampler_index  # convert to int

        if isinstance(index, dotdict): index.index = sampler_index
        else: index = sampler_index

        return dataset_index, index

    @property
    def split(self):
        return self.datasets[0].split

    @property
    def n_views(self):
        return 1

    @property
    def n_latents(self):
        return sum(self.lengths)  # for samplers

    def __getitem__(self, index: Union[dotdict, int]):
        dataset_index, index = self.extract_dataset_index(index)
        dataset = self.datasets[dataset_index]  # get the dataset to sample from
        return dataset.__getitem__(index)
