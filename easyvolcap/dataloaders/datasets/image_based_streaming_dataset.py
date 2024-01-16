import torch
import numpy as np
from typing import List
from scipy import interpolate
from os.path import join, exists, split  # os.path.join is too long

from easyvolcap.engine import DATASETS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.dataloaders.datasets.image_based_dataset import ImageBasedDataset
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.stream_utils import MultiWebcamUSB
from easyvolcap.utils.math_utils import affine_inverse, torch_inverse_3x3, affine_padding
from easyvolcap.utils.cam_utils import generate_hemispherical_orbit, interpolate_camera_path, interpolate_camera_lins, average_c2ws
from easyvolcap.utils.data_utils import UnstructuredTensors, DataSplit, get_rays, get_near_far, as_torch_func, to_tensor, load_resize_undist_ims_bytes

# This dataset should only be used for streaming (demo, no gt loading)
# It is only used by GUI to render novel views of live streaming


@DATASETS.register_module()
class ImageBasedStreamingDataset(ImageBasedDataset):
    def __init__(self,
                 # Dataset intrinsic properties
                 split: str = DataSplit.TEST.name,  # dynamically generated

                 # Stream cameras config
                 stream_cfgs: dotdict = dotdict(),
                 save_images: bool = False,

                 **kwargs,
                 ):
        # Ignore things, since this will serve as a base class of classes supporting *args and **kwargs
        # The inspection of registration and config system only goes down one layer
        # Otherwise it would be to inefficient
        call_from_cfg(super().__init__, kwargs, split=split)
        assert self.closest_using_t == False, "Should not use closest_using_t for streaming dataset"

        # Streaming cameras related configs
        self.stream_cfgs = stream_cfgs
        self.save_images = save_images
        self.open_cameras()  # open all webcams (start streaming)

        # Define the max length of the dataset here, used by `__len__` and `@property`
        self.max_len = 1000000000

    @property
    def frame_min(self): return 0

    @property
    def frame_int(self): return 1  # error out if you call this when giving specific frames

    @property
    def frame_max(self): return self.max_len

    def load_paths(self):
        # Do nothing, since we are streaming
        pass

    def load_vhull(self):
        # Do nothing, since we are streaming
        pass

    def load_bytes(self):
        # Do nothing, since we are streaming
        pass

    def load_smpls(self):
        # Do nothing, since we are streaming
        pass

    def load_bkgds(self):
        # Do nothing, since we are streaming
        pass

    def load_source_indices(self):
        # Do nothing, since we are streaming
        pass

    def __len__(self): return self.max_len  # a super large number

    def t_to_frame(self, t):
        return 0

    def frame_to_latent(self, frame_index):
        return 0

    def open_cameras(self):
        # Open all webcams
        self.streams = MultiWebcamUSB(cam_cfgs=self.stream_cfgs.cam_cfgs,
                                      save_dir=self.stream_cfgs.save_dir,
                                      save_tag=self.stream_cfgs.save_tag)

    def get_sources(self,
                    latent_index: Union[List[int], int], view_index: Union[List[int], int],
                    output: dotdict):
        # Get image from the stream
        imgs = self.streams.capture(save=self.save_images)
        imgs = [to_tensor(imgs[i] / 255.) for i in view_index]  # (S, H, W, 3)
        return imgs

    def get_viewer_batch(self, output: dotdict):
        return ImageBasedDataset.get_viewer_batch(self, output)
