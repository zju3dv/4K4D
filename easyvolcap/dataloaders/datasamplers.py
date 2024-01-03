import torch
import random
import numpy as np

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
from easyvolcap.engine import DATASAMPLERS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, BatchSampler, SequentialSampler, Sampler
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset
from easyvolcap.dataloaders.datasets.volumetric_video_inference_dataset import VolumetricVideoInferenceDataset


@DATASAMPLERS.register_module()
class BatchSampler(BatchSampler):
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int = 8,
                 drop_last: bool = False,
                 *arg,
                 **kwargs,
                 ):
        super().__init__(sampler,
                         batch_size,
                         drop_last)  # strange naming


@DATASAMPLERS.register_module()
class ImageBasedBatchSampler(BatchSampler):
    # This datasampler geneartes a new 'n_imgs' in keys passed into the dataset
    # making sure all dataset instances across different processes returns the same number of
    # source images for collating together a batched input, while maintaining the ability
    # to introduce per-iter randomness in the number of images sampled
    def __init__(self,
                 sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int = 8,
                 drop_last: bool = False,
                 n_srcs_list: List[int] = [2, 3, 4],
                 n_srcs_prob: List[int] = [0.2, 0.6, 0.2],
                 ) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.n_srcs_list = n_srcs_list
        self.n_srcs_prob = n_srcs_prob

    def __iter__(self):
        # Use shared number of images for batching
        iterator = super().__iter__()
        for batch in iterator:
            n_srcs = random.choices(self.n_srcs_list, self.n_srcs_prob)[0]
            batch = [dotdict(index=i, n_srcs=n_srcs) for i in batch]  # expand indices to dotdict
            yield batch


@DATASAMPLERS.register_module()
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler: BatchSampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        self.sampler = self.batch_sampler.sampler

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def get_inds(dataset: VolumetricVideoDataset,
             frame_sample: List[int] = [0, None, 1],
             view_sample: List[int] = [0, None, 1],
             force_manual_frame_selection: bool = False,
             force_manual_view_selection: bool = False,
             ith_latent: int = 0,):
    inds = torch.arange(0, len(dataset))
    nl = 1 if isinstance(dataset, VolumetricVideoInferenceDataset) else dataset.n_latents
    nv = len(dataset) // nl

    # Perform view selection
    view_inds = torch.arange(nv)
    if len(view_sample) != 3 or force_manual_view_selection: view_inds = view_inds[view_sample]  # this is a list of indices
    else: view_inds = view_inds[view_sample[0]:view_sample[1]:view_sample[2]]  # begin, start, end
    if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # Perform frame selection
    frame_inds = torch.arange(nl)
    if len(frame_sample) != 3 or force_manual_frame_selection: frame_inds = frame_inds[frame_sample]
    else: frame_inds = frame_inds[frame_sample[0]:frame_sample[1]:frame_sample[2]]
    if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

    # Actual sampler selection
    inds = inds.reshape(nv, nl)[view_inds][:, frame_inds][:, ith_latent:]
    return inds


@DATASAMPLERS.register_module()
class IterationBasedRandomSampler(RandomSampler):
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 num_samples: int,
                 num_warmups: int = 100000,
                 frame_sample: List[int] = [0, None, 1],
                 view_sample: List[int] = [0, None, 1],
                 ith_latent: int = 0,
                 *arg, **kwargs):
        self.inds = get_inds(dataset, frame_sample, view_sample, ith_latent, *arg, **kwargs).ravel().numpy().tolist()
        super().__init__(data_source=self.inds, num_samples=num_samples)
        self.num_warmups = num_warmups

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        for i in range(self.num_samples):
            if i < self.num_warmups:
                # Randomly sample from range [0, n * i / self.num_warmups)
                idx = random.choice(range(0, max(int(n * i / self.num_warmups), 3)))
            else:
                # Randomly sample from range [0, n)
                idx = random.randrange(n)
            yield self.inds[idx]


# Are there better ways to add existing modules?
@DATASAMPLERS.register_module()
class RandomSampler(RandomSampler):  # cannot use function here, recursion
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 frame_sample: List[int] = [0, None, 1],
                 view_sample: List[int] = [0, None, 1],
                 ith_latent: int = 0,
                 *arg, **kwargs):
        self.inds = get_inds(dataset, frame_sample, view_sample, ith_latent, *arg, **kwargs).ravel().numpy().tolist()
        super().__init__(data_source=self.inds)  # strange naming

    def __iter__(self) -> Iterator[int]:
        yield from [self.inds[i] for i in super().__iter__()]


@DATASAMPLERS.register_module()
class SequentialSampler(SequentialSampler):
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 frame_sample: List[int] = [0, None, 1],
                 view_sample: List[int] = [0, None, 1],
                 ith_latent: int = 0,
                 *arg, **kwargs):
        self.inds = get_inds(dataset, frame_sample, view_sample, ith_latent, *arg, **kwargs).ravel().numpy().tolist()
        super().__init__(data_source=self.inds)  # strange naming

    def __iter__(self) -> Iterator[int]:
        yield from [self.inds[i] for i in super().__iter__()]


@DATASAMPLERS.register_module()
class StreamSampler(SequentialSampler):
    # This is indeed the original sequential sampler
    # Since streaming circumstances only need the sample action, but do not care about the sample content
    # The original sequential sampler and random sampler are both wrapped above, with adiitional
    # `frame_sample` and `view_sample` arguments, which we do not need in streaming
    # So, the best way maybe to wrap the original sequential sampler again?
    def __init__(self,
                 dataset: VolumetricVideoDataset,
                 *arg, **kwargs):
        super().__init__(dataset=dataset, *arg, **kwargs)


DATASAMPLERS.register_module()(DistributedSampler)
