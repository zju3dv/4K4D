import torch
import numpy as np
from typing import List
from scipy import interpolate

from easyvolcap.engine import DATASETS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.dataloaders.datasets.volumetric_video_dataset import VolumetricVideoDataset

from easyvolcap.engine import cfg, args  # global
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera, to_easymocap, write_camera
from easyvolcap.utils.math_utils import affine_inverse, torch_inverse_3x3, affine_padding
from easyvolcap.utils.data_utils import get_rays, DataSplit, get_near_far, as_torch_func, export_camera
from easyvolcap.utils.cam_utils import generate_hemispherical_orbit, interpolate_camera_path, interpolate_camera_lins, generate_spiral_path, Interpolation

# This dataset should only be used for inference (no gt loading)
# It also gives user the option to activate playback during rendering
# By default, it will generate a spiral path from specified extri and intri
# It can also read predefined custom camera path (possibly defined in GUI)


@DATASETS.register_module()
class VolumetricVideoInferenceDataset(VolumetricVideoDataset):
    def __init__(self,
                 # Other default configurations
                 n_render_views: int = 300,  # number of frames to render, 5s for 60fps

                 # Hemisphere or custom camera path
                 interp_type: str = Interpolation.ORBIT.name,  # Interpolation.CUBIC or Interpolation.LINEAR
                 interp_cfg: dotdict = dotdict(
                     orbit_radius=-1,  # if < 0, will use the avearge radius
                     orbit_height=0.,  # sphere_height shift
                     smoothing_term=-1.0,  # negativa values -> compute spiral path, otherwise just interpolate
                 ),

                 # 0-1 or ranges from camera path or specific
                 temporal_range: List[float] = [0, 1],  # will linearly interpolate this (if empty, use custom)

                 interp_using_t: bool = False,  # whether to interpolate along time dimension of camera parameters
                 render_size_default: List[int] = [1080, 1920],  # when no H, W in intri.yml # MARK: updated default
                 render_size: List[int] = [-1, -1],  # rendering size
                 focal_ratio: float = 1.0,  # use the H, W to determine intrinsics

                 # If these files are present, will not perform interpolation from training poses, but simply render the given path
                 camera_path_intri: str = None,  # path to intri.yml
                 camera_path_extri: str = None,  # path to extri.yml

                 save_interp_path: bool = True,  # save the interpolated path, find it is useful too
                 render_path_root: str = 'data/novel_view',  # root path for saving the interpolated render path
                 **kwargs,
                 ):
        # NOTE: no super().__init__() since these datasets are fundamentally different
        # We only want to reuse some of the functionalities
        call_from_cfg(super().__init__, kwargs)

        self.camera_path_intri = camera_path_intri
        self.camera_path_extri = camera_path_extri

        self.n_render_views = n_render_views  # this variable is required for init
        self.interp_type = Interpolation[interp_type]
        self.interp_cfg = interp_cfg
        self.interp_using_t = interp_using_t
        self.temporal_range_overwrite = temporal_range if not interp_using_t else [0]
        self.input_Ks = self.Ks.clone()  # for exporting camera paths
        self.input_c2ws = self.c2ws.clone()
        self.load_interpolations()

        self.render_size_default = render_size_default
        self.render_size = render_size
        self.focal_ratio = focal_ratio
        self.load_default_sizes()

        self.save_interp_path = save_interp_path
        self.render_path_root = render_path_root
        self.save_interpolated_path()

    def load_paths(self):
        if self.use_vhulls:
            super().load_paths()

    def load_bytes(self):
        # Only update intrinsics
        self.Ks = self.Ks.clone()  # clone before every inplace operation to avoid bugs
        self.Ks[..., :2, :] *= self.ratio
        self.Hs = (self.Hs * self.ratio).int()
        self.Ws = (self.Ws * self.ratio).int()

        # Load actual visual hulls
        if self.use_vhulls:
            super().load_vhulls()

    def load_default_sizes(self):
        self.load_size(self.render_size, overwrite=True)  # when overwritting, will change K
        self.load_size(self.render_size_default)

    def load_size(self, size: List[int], overwrite: bool = False):
        if size[0] < 0: return
        # Prepare default size
        H, W = size
        M = max(H, W)
        K = torch.as_tensor([
            [self.focal_ratio * M, 0, W / 2],  # smaller focal, large fov for a bigger picture
            [0, self.focal_ratio * M, H / 2],
            [0, 0, 1],
        ], dtype=torch.float)

        # Load the default image size
        sh = self.Ks.shape
        self.Ks, self.Hs, self.Ws = self.Ks.reshape(-1, 3, 3), self.Hs.reshape(-1), self.Ws.reshape(-1)
        if not overwrite:
            self.Ks = torch.stack([(k if h > 0 and w > 0 else K) for k, h, w in zip(self.Ks, self.Hs, self.Ws)])  # N, 3, 3
            self.Hs = torch.as_tensor([(h if h > 0 else H) for h in self.Hs])
            self.Ws = torch.as_tensor([(w if w > 0 else W) for w in self.Ws])
        else:
            self.Ks = K[None].expand(len(self.Ks), -1, -1)
            self.Hs = torch.full_like(self.Hs, H)
            self.Ws = torch.full_like(self.Ws, W)
        self.Ks, self.Hs, self.Ws = self.Ks.view(sh), self.Hs.view(sh[:-2]), self.Ws.view(sh[:-2])

    def load_interpolations(self):
        if self.camera_path_intri and self.camera_path_extri:
            # For image based dataset, the original camera parameters has already been backed up in src_exts and src_ixts
            log(yellow(f'Will use camera path from {blue(self.camera_path_extri)} and {blue(self.camera_path_intri)} to interpolate the camera curve'))
            old_cameras = self.cameras
            old_intri_file = self.intri_file
            old_extri_file = self.extri_file
            old_view_sample = self.view_sample
            old_use_aligned_cameras = self.use_aligned_cameras

            self.view_sample = [0, None, 1]  # should select all views since we want to interpolate between or just use all of them
            self.use_aligned_cameras = False
            self.intri_file = os.path.abspath(self.camera_path_intri)  # there's a join operation with data_root, using abspath uses this path
            self.extri_file = os.path.abspath(self.camera_path_extri)  # there's a join operation with data_root, using abspath uses this path

        self.load_cameras()
        self.select_cameras()
        self.interpolate_extrinsics()
        self.interpolate_intrinsics()
        self.interpolate_times()

        if self.camera_path_intri and self.camera_path_extri:
            self.use_aligned_cameras = old_use_aligned_cameras
            self.view_sample = old_view_sample
            self.intri_file = old_intri_file
            self.extri_file = old_extri_file
            self.cameras = old_cameras

    def interpolate_extrinsics(self):
        # Load the extrinsic parameters
        if self.interp_using_t: self.c2ws = self.c2ws[0]
        else: self.c2ws = self.c2ws[:, 0]

        if self.interp_type == Interpolation.CUBIC:
            self.c2ws = as_torch_func(interpolate_camera_path)(self.c2ws, self.n_render_views, **self.interp_cfg)
        elif self.interp_type == Interpolation.ORBIT:
            self.c2ws = as_torch_func(generate_hemispherical_orbit)(self.c2ws, self.n_render_views, **self.interp_cfg)
        elif self.interp_type == Interpolation.SPIRAL:
            self.c2ws = as_torch_func(generate_spiral_path)(self.c2ws, self.n_render_views, **self.interp_cfg)
        elif self.interp_type == Interpolation.SECTOR:
            pass  # TODO: Implement this
        elif self.interp_type == Interpolation.NONE:
            if len(self.c2ws) != self.n_render_views:
                log(yellow(f'The number of views in the camera path ({blue(len(self.c2ws))}) does not match the number of views to render ({blue(self.n_render_views)}), will use the one in the camera path ({blue(len(self.c2ws))})'))
            self.n_render_views = len(self.c2ws)
        else:
            raise NotImplementedError

        # Add skeleton dimension for reusing camera parameter loading mechanism
        self.c2ws = self.c2ws[:, None].expand(-1, self.n_latents, -1, -1)  # (V, F, 3, 4)
        self.w2cs = affine_inverse(self.c2ws)  # (V, F, 3, 4)
        self.Rs = self.w2cs[..., :-1]  # (V, F, 3, 3)
        self.Ts = self.w2cs[..., -1:]  # (V, F, 3, 1)
        self.Cs = self.c2ws[..., -1:]  # updated camera center

        # After interpolation, the first dim of all cameras should match the actual number of dims to render
        # The second dim should be 1 in this implementation

    def interpolate_intrinsics(self):
        # Interpolate the intrinsic parameters
        if self.interp_using_t: self.Ks, self.Hs, self.Ws, self.ns, self.fs = self.Ks[0], self.Hs[0], self.Ws[0], self.ns[0], self.fs[0]
        else: self.Ks, self.Hs, self.Ws, self.ns, self.fs = self.Ks[:, 0], self.Hs[:, 0], self.Ws[:, 0], self.ns[:, 0], self.fs[:, 0]

        if self.interp_type != Interpolation.NONE:

            self.Ks = torch.as_tensor(interpolate_camera_lins(self.Ks.float().view(-1, 9).numpy(), self.n_render_views, **self.interp_cfg)).view(-1, 3, 3)
            self.Hs = torch.as_tensor(interpolate_camera_lins(self.Hs.float().view(-1, 1).numpy(), self.n_render_views, **self.interp_cfg)).view(-1).int()
            self.Ws = torch.as_tensor(interpolate_camera_lins(self.Ws.float().view(-1, 1).numpy(), self.n_render_views, **self.interp_cfg)).view(-1).int()
            self.ns = torch.as_tensor(interpolate_camera_lins(self.ns.float().view(-1, 1).numpy(), self.n_render_views, **self.interp_cfg)).view(-1)
            self.fs = torch.as_tensor(interpolate_camera_lins(self.fs.float().view(-1, 1).numpy(), self.n_render_views, **self.interp_cfg)).view(-1)

        # Add skeleton dimension for reusing camera parameter loading mechanism
        self.Ks = self.Ks[:, None].expand(-1, self.n_latents, -1, -1)  # (V, F, 3, 3)
        self.Hs = self.Hs[:, None].expand(-1, self.n_latents).int()  # (V, F)
        self.Ws = self.Ws[:, None].expand(-1, self.n_latents).int()  # (V, F)
        self.ns = self.ns[:, None].expand(-1, self.n_latents)  # (V, F)
        self.fs = self.fs[:, None].expand(-1, self.n_latents)  # (V, F)

    def interpolate_times(self):
        # Interpolate the temporal parameters

        # We'd like to set temporal range 0-1 for rendering all loaded frames
        # However the t=1 frame is not loaded due to 0->999 range of python index (contrast to 1-1000)
        # This results in an error when rendering the last frame, thus frame_max gets a -1
        if self.temporal_range_overwrite is not None and len(self.temporal_range_overwrite):
            ts = torch.as_tensor(self.temporal_range_overwrite).float()  # 2, will this work?
        else:
            ts = self.ts[0] if self.interp_using_t else self.ts[:, 0]  # only taking the temporal range of the first view, we assume they are shared

        if len(ts) > 1: self.ts = torch.as_tensor(interpolate_camera_lins(ts.view(-1, 1).numpy(), self.n_render_views, **self.interp_cfg)).view(-1)[:, None].expand(-1, self.n_latents)  # (V, F)
        else: self.ts = torch.as_tensor(ts).repeat(self.n_render_views)[:, None].expand(-1, self.n_latents)  # 100, (V, F)

    def save_interpolated_path(self):
        # Directly return if not saving
        if not self.save_interp_path: return

        if self.render_path_root is None:
            if self.camera_path_intri is not None:
                self.render_path_root = dirname(self.camera_path_intri)
        else:
            try: save_tag = cfg.runner_cfg.visualizer_cfg.save_tag  # MARK: GLOBAL
            except: save_tag = ''
            self.render_path_root = join(self.render_path_root, cfg.exp_name)  # MARK: GLOBAL
            if save_tag != '': self.render_path_root = join(self.render_path_root, str(save_tag))

        # Save the interpolated paths otherwise
        render_path_cams = to_easymocap(self.Ks, self.Hs, self.Ws, self.Rs, self.Ts, self.ts, self.ns, self.fs)
        write_camera(render_path_cams, self.render_path_root)

        # Save the visualized camera paths
        export_camera(self.input_c2ws[:, 0], self.input_Ks[:, 0], filename=join(self.render_path_root, 'input_camera.ply'))
        export_camera(self.c2ws[:, 0], self.Ks[:, 0], filename=join(self.render_path_root, 'interp_camera.ply'))

    def get_indices(self, index: int):
        view_index = index
        camera_index = index
        t = self.ts[view_index][0]

        frame_index = self.t_to_frame(t)
        latent_index = self.frame_to_latent(frame_index)
        return view_index, latent_index, camera_index, frame_index

    def __getitem__(self, index: int):
        return self.get_metadata(index)

    # MARK: Number of frames is actually the same as number of views
    def __len__(self):
        return self.n_render_views
