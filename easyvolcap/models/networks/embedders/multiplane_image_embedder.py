import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import EMBEDDERS, REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import export_pts
from easyvolcap.utils.net_utils import interpolate_image
from easyvolcap.utils.mpi_utils import StereoMagnificationNet, raw2mpi, render_debug_plane_sweep_volume


@EMBEDDERS.register_module()
class MultiplaneImageEmbedder(nn.Module):
    def __init__(self,
                 n_planes: int = 32,  # 
                 smf_cfg: dotdict = dotdict(type=StereoMagnificationNet.__name__),
                 ) -> None:
        super().__init__()
        self.smf = REGRESSORS.build(smf_cfg, n_planes=n_planes)
        self.img_pad = self.smf.size_pad
        self.out_dim = 4  # rgba, FIXME: determine whether it is a good idea to hard code this
                          # Or, in other word, whether invoking `raw2mpi` here is a good idea or not

    def forward(self, xyz: torch.Tensor, batch: dotdict):
        # Fetch useful data from batch
        source_vol = batch.output.source_vol  # (B, 3, D, H, W)
        ref_inp = batch.ref_inp  # (B, 3, H, W)
        tar2ref_grid = batch.output.tar2ref_grid  # (B, D, H, W, 2)

        # Deal with nasty batch dimension
        B, _, _, H, W = source_vol.shape
        # Preparing source scaling (for painless up convolution and skip connections)
        Hp, Wp = int(np.ceil(H / self.img_pad)) * self.img_pad, int(np.ceil(W / self.img_pad)) * self.img_pad  # Input and output should be same in size
        source_vol = interpolate_image(source_vol.permute(0, 2, 1, 3, 4), size=(Hp, Wp)).permute(0, 2, 1, 3, 4)  # (B, 3, D, Hp, Wp)
        ref_inp = interpolate_image(ref_inp, size=(Hp, Wp))  # (B, 3, H, W) -> (B, 3, Hp, Wp)
        sfm_inputs = torch.cat([ref_inp, source_vol.reshape(B, -1, Hp, Wp)], dim=1)  # (B, 3 + 3 * D, Hp, Wp)

        # Compute and parse network output
        # 1. [*]1 unified background image: (B, 3, Hp, Wp), the predicted background image, shared by all depth planes
        # 2. [*]D alpha images: (B, D, Hp, Wp), the predicted alpha images for each depth plane
        # 3. [*]D blend weights: (B, D, Hp, Wp), the predicted blend weights for each depth plane
        sfm_output = self.smf(sfm_inputs)  # (B, 3 + 2 * D, Hp, Wp)
        # Build the multiplane image from the raw network output
        sfm_output = interpolate_image(sfm_output, size=(H, W))  # (B, 3 + 2 * D, H, W)
        ref_mpi_rgba = raw2mpi(sfm_output, batch)  # (B, H, W, D, 4)

        # # debug reference view mpi construction
        # export_pts(xyz, color=ref_mpi_rgba[..., :3].reshape(B, -1, 3), filename='ref_mpi.ply')
        # __import__('easyvolcap.utils.console_utils', fromlist=['debugger']).debugger()

        # 1d grid sampling, (B, H, W, D, 4) -> (B, D, 4, H, W) -> (B * D, 4, H, W)
        ref_mpi_rgba = ref_mpi_rgba.permute(0, 3, 4, 1, 2).reshape(-1, 4, H, W)
        tar2ref_grid = tar2ref_grid.reshape(-1, H, W, 2)  # (B * D, H, W, 2)
        tar_mpi_rgba = F.grid_sample(ref_mpi_rgba, tar2ref_grid, padding_mode='border')  # (B * D, 4, H, W)
        # (B * D, 4, H, W) -> (B, D, 4, H, W) -> (B, H, W, D, 4)
        tar_mpi_rgba = tar_mpi_rgba.reshape(B, -1, 4, H, W).permute(0, 3, 4, 1, 2)

        # # debug target view mpi construction
        # export_pts(xyz, color=tar_mpi_rgba[..., :3].reshape(B, -1, 3), filename='tar_mpi.ply')
        # __import__('easyvolcap.utils.console_utils', fromlist=['debugger']).debugger()

        return tar_mpi_rgba.reshape(B, -1, 4)
