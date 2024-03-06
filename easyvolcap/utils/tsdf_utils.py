"""
Copied from: https://github.com/nianticlabs/simplerecon/blob/main/tools/tsdf.py
"""

import os
import torch
import trimesh
import numpy as np
import torch.nn.functional as F
from skimage import measure
from typing import Tuple


class TSDF:

    """
        Class for housing and data handling TSDF volumes.
    """
    # Ensures the final voxel volume dimensions are multiples of 8
    VOX_MOD = 8

    def __init__(
        self,
        voxel_coords: torch.Tensor,
        tsdf_values: torch.Tensor,
        tsdf_weights: torch.Tensor,
        voxel_size: float,
        origin: torch.Tensor,
    ):
        """
            Sets interal class attributes.
        """
        self.voxel_coords = voxel_coords.half()
        self.tsdf_values = tsdf_values.half()
        self.tsdf_weights = tsdf_weights.half()
        self.voxel_size = voxel_size
        self.origin = origin.half()

    @classmethod
    def from_file(cls, tsdf_file):
        """ Loads a tsdf from a numpy file. """
        tsdf_data = np.load(tsdf_file)

        tsdf_values = torch.from_numpy(tsdf_data['tsdf_values'])
        origin = torch.from_numpy(tsdf_data['origin'])
        voxel_size = tsdf_data['voxel_size'].item()

        tsdf_weights = torch.ones_like(tsdf_values)

        voxel_coords = cls.generate_voxel_coords(origin, tsdf_values.shape[1:], voxel_size)

        return TSDF(voxel_coords, tsdf_values, tsdf_weights, voxel_size)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, voxel_size: float):
        """ Gets TSDF bounds from a mesh file. """
        xmax, ymax, zmax = mesh.vertices.max(0)
        xmin, ymin, zmin = mesh.vertices.min(0)

        bounds = {'xmin': xmin, 'xmax': xmax,
                  'ymin': ymin, 'ymax': ymax,
                  'zmin': zmin, 'zmax': zmax}

        # create a buffer around bounds
        for key, val in bounds.items():
            if 'min' in key:
                bounds[key] = val - 3 * voxel_size
            else:
                bounds[key] = val + 3 * voxel_size
        return cls.from_bounds(bounds, voxel_size)

    @classmethod
    def from_bounds(cls, bounds: dict, voxel_size: float):
        """ Creates a TSDF volume with bounds at a specific voxel size. """

        expected_keys = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for key in expected_keys:
            if key not in bounds.keys():
                raise KeyError("Provided bounds dict need to have keys"
                               "'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'!")

        num_voxels_x = int(
            np.ceil((bounds['xmax'] - bounds['xmin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        num_voxels_y = int(
            np.ceil((bounds['ymax'] - bounds['ymin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        num_voxels_z = int(
            np.ceil((bounds['zmax'] - bounds['zmin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD

        origin = torch.FloatTensor([bounds['xmin'], bounds['ymin'], bounds['zmin']])

        voxel_coords = cls.generate_voxel_coords(
            origin, (num_voxels_x, num_voxels_y, num_voxels_z), voxel_size).half()

        # init to -1s
        tsdf_values = -torch.ones_like(voxel_coords[0]).half()
        tsdf_weights = torch.zeros_like(voxel_coords[0]).half()

        return TSDF(voxel_coords, tsdf_values, tsdf_weights, voxel_size, origin)

    @classmethod
    def generate_voxel_coords(cls,
                              origin: torch.Tensor,
                              volume_dims: Tuple[int, int, int],
                              voxel_size: float):
        """ Gets world coordinates for each location in the TSDF. """

        grid = torch.meshgrid([torch.arange(vd) for vd in volume_dims])

        voxel_coords = origin.view(3, 1, 1, 1) + torch.stack(grid, 0) * voxel_size

        return voxel_coords

    def cuda(self):
        """ Moves TSDF to gpu memory. """
        self.voxel_coords = self.voxel_coords.cuda()
        self.tsdf_values = self.tsdf_values.cuda()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cuda()

    def cpu(self):
        """ Moves TSDF to cpu memory. """
        self.voxel_coords = self.voxel_coords.cpu()
        self.tsdf_values = self.tsdf_values.cpu()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cpu()

    def to_mesh(self, scale_to_world=True, export_single_mesh=False):
        """ Extracts a mesh from the TSDF volume using marching cubes. 

            Args:
                scale_to_world: should we scale vertices from TSDF voxel coords 
                    to world coordinates?
                export_single_mesh: returns a single walled mesh from marching
                    cubes. Requires a custom implementation of 
                    measure.marching_cubes that supports single_mesh

        """
        tsdf = self.tsdf_values.detach().cpu().clone().float()
        tsdf_np = tsdf.clamp(-1, 1).cpu().numpy()

        if export_single_mesh:
            verts, faces, norms, _ = measure.marching_cubes(
                tsdf_np,
                level=0,
                allow_degenerate=False,
                single_mesh=True,
            )
        else:
            verts, faces, norms, _ = measure.marching_cubes(
                tsdf_np,
                level=0,
                allow_degenerate=False,
            )

        if scale_to_world:
            verts = self.origin.cpu().view(1, 3) + verts * self.voxel_size

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=norms)
        return mesh

    def save(self, savepath, filename, save_mesh=True):
        """ Saves a mesh to disk. """
        self.cpu()
        os.makedirs(savepath, exist_ok=True)

        if save_mesh:
            mesh = self.to_mesh()
            trimesh.exchange.export.export_mesh(
                mesh, os.path.join(savepath,
                                   filename).replace(".bin", ".ply"), "ply")


class TSDFFuser:
    """
        Class for fusing depth maps into TSDF volumes.
    """

    def __init__(self, tsdf, min_depth=0.5, max_depth=5.0, use_gpu=True):
        """
            Inits the fuser with fusing parameters.

            Args:
                tsdf: a TSDF volume object.
                min_depth: minimum depth to limit inomcing depth maps to.
                max_depth: maximum depth to limit inomcing depth maps to.
                use_gpu: use cuda?

        """
        self.tsdf = tsdf
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_gpu = use_gpu
        self.truncation_size = 3.0
        self.maxW = 100.0

        # Create homogeneous coords once only
        self.hom_voxel_coords_14hwd = torch.cat(
            (self.voxel_coords, torch.ones_like(self.voxel_coords[:1])), 0).unsqueeze(0)

    @property
    def voxel_coords(self):
        return self.tsdf.voxel_coords

    @property
    def tsdf_values(self):
        return self.tsdf.tsdf_values

    @property
    def tsdf_weights(self):
        return self.tsdf.tsdf_weights

    @property
    def voxel_size(self):
        return self.tsdf.voxel_size

    @property
    def shape(self):
        return self.voxel_coords.shape[1:]

    @property
    def truncation(self):
        return self.truncation_size * self.voxel_size

    def project_to_camera(self, cam_T_world_T_b44, K_b44):

        if self.use_gpu:
            cam_T_world_T_b44 = cam_T_world_T_b44.cuda()
            K_b44 = K_b44.cuda()
            self.hom_voxel_coords_14hwd = self.hom_voxel_coords_14hwd.cuda()

        world_to_pix_P_b34 = torch.matmul(K_b44, cam_T_world_T_b44)[:, :3]
        batch_size = cam_T_world_T_b44.shape[0]

        world_points_b4N = \
            self.hom_voxel_coords_14hwd.expand(batch_size, 4, *self.shape).flatten(start_dim=2)
        cam_points_b3N = torch.matmul(world_to_pix_P_b34, world_points_b4N)
        cam_points_b3N[:, :2] = cam_points_b3N[:, :2] / cam_points_b3N[:, 2, None]

        return cam_points_b3N

    def integrate_depth(
        self,
        depth_b1hw,
        cam_T_world_T_b44,
        K_b44,
        depth_mask_b1hw=None,
    ):
        """
            Integrates depth maps into the volume. Supports batching.

            depth_b1hw: tensor with depth map
            cam_T_world_T_b44: camera extrinsics (not pose!).
            K_b44: camera intrinsics.
            depth_mask_b1hw: an optional boolean mask for valid depth points in 
                the depth map. 
        """
        img_h, img_w = depth_b1hw.shape[2:]
        img_size = torch.Tensor([img_w, img_h], dtype=torch.float16).view(1, 1, 1, 2)
        if self.use_gpu:
            depth_b1hw = depth_b1hw.cuda()
            img_size = img_size.cuda()
            self.tsdf.cuda()

        # Project voxel coordinates into images
        cam_points_b3N = self.project_to_camera(cam_T_world_T_b44, K_b44)
        vox_depth_b1N = cam_points_b3N[:, 2:3]
        pixel_coords_b2N = cam_points_b3N[:, :2]

        # Reshape the projected voxel coords to a 2D view of shape Hx(WxD)
        pixel_coords_bhw2 = pixel_coords_b2N.view(-1, 2, self.shape[0],
                                                  self.shape[1] * self.shape[2]
                                                  ).permute(0, 2, 3, 1)
        pixel_coords_bhw2 = 2 * pixel_coords_bhw2 / img_size - 1

        if depth_mask_b1hw is not None:
            depth_b1hw = depth_b1hw.clone()
            depth_b1hw[~depth_mask_b1hw] = -1

        # Sample the depth using grid sample
        sampled_depth_b1hw = F.grid_sample(input=depth_b1hw,
                                           grid=pixel_coords_bhw2,
                                           mode="nearest",
                                           padding_mode="zeros",
                                           align_corners=False)
        sampled_depth_b1N = sampled_depth_b1hw.flatten(start_dim=2)

        # Confidence from InfiniTAM
        confidence_b1N = torch.clamp(
            1.0 - (sampled_depth_b1N - self.min_depth) / (self.max_depth - self.min_depth),
            min=0.0, max=1.0) ** 2

        # Calculate TSDF values from depth difference by normalizing to [-1, 1]
        dist_b1N = sampled_depth_b1N - vox_depth_b1N
        tsdf_vals_b1N = torch.clamp(dist_b1N / self.truncation, min=-1.0, max=1.0)

        # Get the valid points mask
        valid_points_b1N = (vox_depth_b1N > 0) & (dist_b1N > -self.truncation) & \
            (sampled_depth_b1N > 0) & (vox_depth_b1N > 0) & (vox_depth_b1N < self.max_depth) & \
            (confidence_b1N > 0)

        # Updating the TSDF has to be sequential so we break out the batch here
        for tsdf_val_1N, valid_points_1N, confidence_1N in zip(tsdf_vals_b1N,
                                                               valid_points_b1N,
                                                               confidence_b1N):
            # Reshape the valid mask to the TSDF's shape and read the old values
            valid_points_hwd = valid_points_1N.view(self.shape)
            old_tsdf_vals = self.tsdf_values[valid_points_hwd]
            old_weights = self.tsdf_weights[valid_points_hwd]

            # Fetch the new tsdf values and the confidence
            new_tsdf_vals = tsdf_val_1N[valid_points_1N]
            confidence = confidence_1N[valid_points_1N]

            # More infiniTAM magic: update faster when the new samples are more confident
            update_rate = torch.where(confidence < old_weights, 2.0, 5.0).half()

            # Compute the new weight and the normalization factor
            new_weights = confidence * update_rate / self.maxW
            total_weights = old_weights + new_weights

            # Update the tsdf and the weights
            self.tsdf_values[valid_points_hwd] = (old_tsdf_vals * old_weights + new_tsdf_vals * new_weights) / total_weights
            self.tsdf_weights[valid_points_hwd] = torch.clamp(total_weights, max=1.0)
