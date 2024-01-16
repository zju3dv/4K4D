"""
# This dataset creates a volume to evaluate the network in a 3D grid
# B, P, C
xyz: torch.Tensor = batch.xyz
dir: torch.Tensor = batch.dir
t: torch.Tensor = batch.t
dist: torch.Tensor = batch.dist
"""

import torch
from easyvolcap.engine import DATASETS
from easyvolcap.engine.registry import call_from_cfg

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.vhull_utils import hierarchically_carve_vhull
from easyvolcap.utils.data_utils import add_batch, to_tensor, to_cuda, to_cpu, load_image_from_bytes
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset


@DATASETS.register_module()
class GeometryDataset(VolumetricVideoDataset):
    def __init__(self,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs)

        def carve_using_bytes(H, W, K, R, T, latent_index):
            if hasattr(self, 'mks_bytes'):
                bytes = [self.mks_bytes[i * self.n_latents + latent_index] for i in range(len(H))]  # get mask bytes of this frame
                msks = parallel_execution(bytes, normalize=True, action=load_image_from_bytes, sequential=True)
                msks = to_tensor(msks)

                # Fill blank canvas for each mask
                # It should be OK to use black images without resizing since we're performing grid_sampling
                N = len(msks)
                H_max = max([msk.shape[-3] for msk in msks])
                W_max = max([msk.shape[-2] for msk in msks])
                msks_full = H.new_zeros(N, H_max, W_max, 1, dtype=torch.float)
                for i, (h, w, msk) in enumerate(zip(H, W, msks)):
                    msks_full[i, :h, :w, :] = msk  # fill
                msks = msks_full  # N, H, W, 1
            else:
                msks = torch.ones(len(H), H[0].item(), W[0].item(), 1, dtype=torch.float)  # N, H, W, 1

            # Actual carving
            inputs = H, W, K, R, T, msks, self.get_bounds(latent_index)  # this is the starting bounds
            inputs = to_cpu(inputs)
            vhulls, bounds, valid, inds = hierarchically_carve_vhull(
                *inputs,
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

        self.xyz, self.input_bounds, self.valid, self.inds = [], [], [], []
        for i in tqdm(range(self.n_latents), desc='Preparing input xyz to extract geometry'):
            xyz, bounds, valid, inds = carve_using_bytes(
                self.Hs[:, i],
                self.Ws[:, i],
                self.Ks[:, i],
                self.Rs[:, i],
                self.Ts[:, i],
                i,
            )
            self.xyz.append(xyz)
            self.input_bounds.append(bounds)
            self.valid.append(valid)
            self.inds.append(inds)

    def __getitem__(self, index: int):
        output = self.get_metadata(index)
        output.xyz = self.xyz[output.latent_index]
        output.inds = self.inds[output.latent_index]
        output.valid = self.valid[output.latent_index]
        output.bounds = self.input_bounds[output.latent_index]

        output.W, output.H, output.D = output.valid.shape[:3]
        output.meta.W, output.meta.H, output.meta.D = output.W, output.H, output.D
        output.meta.voxel_size = torch.as_tensor(self.vhull_voxel_size, dtype=torch.float)

        output.dir = normalize(-output.xyz)  # point inward to the origin
        output.dist = torch.full_like(output.xyz[..., :1], self.vhull_voxel_size)
        return output
