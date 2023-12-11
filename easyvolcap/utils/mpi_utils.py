import torch

from torch import nn
from typing import Union, List
from torch.nn import functional as F

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import export_pts, export_pcd, export_mesh
from easyvolcap.utils.net_utils import create_meshgrid, affine_inverse, normalize, interpolate_image, linear_sampling, s_vals_to_z_vals, ray2xyz, get_rays


def get_proj_mats(src_scale: Union[List, torch.Tensor], tar_scale: Union[List, torch.Tensor],
                  tar_ext: torch.Tensor, tar_ixt: torch.Tensor,
                  src_exts: torch.Tensor, src_ixts: torch.Tensor,
                  ):
    # For original multiplane image (MPI) representation, there is only one source image
    # See https://github.com/DongGeun-Yoon/Stereo-Magnification-Learning-view-synthesis-using-multiplane-images-MPIs-/blob/master/dataset.py
    # But for modified MPI, like deepview, there are multiple source images
    # See https://github.com/Findeton/deepview/blob/main/dset_realestate/dset1.py#L164
    if src_exts.ndim < 4 and src_ixts.ndim < 4:
        src_exts = src_exts.unsqueeze(dim=-3)  # (B, S, 4, 4) == (B, 1, 4, 4)
        src_ixts = src_ixts.unsqueeze(dim=-3)  # (B, S, 3, 3) == (B, 1, 3, 3)

    if isinstance(src_scale, List): src_scale = torch.tensor(src_scale, dtype=torch.float32, device=tar_ext.device)[..., None]
    if isinstance(tar_scale, List): tar_scale = torch.tensor(tar_scale, dtype=torch.float32, device=tar_ext.device)[..., None]

    # It's possible for H W to have difference actual scale
    # The configurable scale is only a guidance, since convolutions asks for 4x
    src_ixts = src_ixts.clone()  # (B, S, 3, 3)
    src_ixts[..., :2, :] *= src_scale  # (B, S, 3, 3)
    src_projs = src_ixts @ src_exts[..., :3, :]  # (B, S, 3, 3) @ (B, S, 3, 4) -> (B, S, 3, 4)

    tar_ixt = tar_ixt.clone()  # (B, 3, 3)
    tar_ixt[..., :2, :] *= tar_scale  # (B, 3, 3)
    tar_projs = tar_ixt @ tar_ext[..., :3, :]  # (B, 3, 4)

    # Homography coordinate
    tar_ones = torch.zeros_like(tar_ext[..., -1:, :])  # (B, 1, 4)
    tar_ones[..., -1] = 1  # (B, 1, 4)
    tar_projs = torch.cat((tar_projs, tar_ones), dim=-2)  # (B, 4, 4)
    tar_projs_inv = torch.inverse(tar_projs)  # (B, 4, 4)

    # (B, S, 3, 4) @ (B, 1, 4, 4) -> (B, S, 3, 4)
    proj_mats = src_projs @ tar_projs_inv[..., None, :, :]
    return proj_mats


def get_rela_mats(tar_ext: torch.Tensor, src_ext: torch.Tensor):
    return tar_ext @ torch.inverse(src_ext)  # (B, 4, 4)


def divide_safe_torch(num, den):
    # use torch.finfo to avoid divide by zero
    eps = torch.finfo(num.dtype).eps
    # substitute 0 with eps
    den = den + eps * (den == 0).float()
    return torch.div(num, den)


def get_inverse_homography(src_ixt: torch.Tensor, tar_ixt: torch.Tensor,
                           R: torch.Tensor, t: torch.Tensor,
                           n_hat: torch.Tensor, a: torch.Tensor):
    """Get inverse homography matrix from target image plane to source camera depth planes.
    Args:
        src_ixt (torch.Tensor): (D, B, 3, 3), intrinsic matrix of source camera.
        tar_ixt (torch.Tensor): (D, B, 3, 3), intrinsic matrix of target camera.
        R (torch.Tensor): (D, B, 3, 3), rotation matrix from source camera to target camera.
        t (torch.Tensor): (D, B, 3, 1), translation vector from source camera to target camera.
        n_hat (torch.Tensor): (D, B, 1, 3), normal vector of the plane.
        a (torch.Tensor): (D, B, 1, 1), depth of the plane.
    Returns:
        inv_homo (torch.Tensor): (D, B, 3, 3), homography matrix from source camera to target camera.
    """
    # See section 3.3 in https://arxiv.org/abs/1805.09817 for more detail
    R_trans = torch.transpose(R, dim0=-2, dim1=-1)  # (D, B, 3, 3)
    denom = a - torch.matmul(torch.matmul(n_hat, R_trans), t)  # (D, B, 1, 1)
    numer = torch.matmul(torch.matmul(torch.matmul(R_trans, t), n_hat), R_trans)  # (D, B, 3, 3)
    # Compute the final inverse homography transformation matrix
    inv_homo = torch.matmul(src_ixt, torch.matmul(R_trans + divide_safe_torch(numer, denom), torch.inverse(tar_ixt)))
    return inv_homo


def inverse_warpping(depth: torch.Tensor, mats: torch.Tensor, tar_ixt: torch.Tensor, src_ixt: torch.Tensor):
    """Inverse warp target image pixels to reference sweep volume planes.
    Args:
        depth (torch.Tensor): (B, D, H, W), sampled sweep volume depth of each plane, in reference camera coordinate.
        mats (torch.Tensor): (B, 4, 4), relative transformation matrix from reference camera coordinate to target camera coordinate.
        tar_ixt (torch.Tensor): (B, 3, 3), intrinsic matrix of target camera.
        src_ixt (torch.Tensor): (B, 3, 3), intrinsic matrix of reference camera.
    Returns:
        warpped_pts (torch.Tensor): (B, D, H * W, 2), the warpped target image pixels in reference image.
    """
    # Prepare shapes
    B, D, H, W = depth.shape
    a = -depth[..., :1, :1].permute(1, 0, 2, 3)  # (D, B, 1, 1)

    # Prepare intrinsic matrix
    tar_ixt = tar_ixt[None].expand(D, -1, -1, -1)  # (D, B, 3, 3)
    src_ixt = src_ixt[None].expand(D, -1, -1, -1)  # (D, B, 3, 3)

    # Prepare R, t
    mats = mats[None].expand(D, -1, -1, -1)  # (D, B, 4, 4), no memory allocation
    R = mats[..., :3, :3]  # (D, B, 3, 3)
    t = mats[..., :3, 3:]  # (D, B, 3, 1)

    # Prepare plane normal w.r.t source camera frame (typically [0 0 1])
    n_hat = torch.tensor([0, 0, 1], dtype=torch.float32, device=depth.device)  # (3,)
    n_hat = n_hat[None, None, None].expand(D, B, -1, -1)  # (D, B, 1, 3)
    
    # Construct inverse homography matrix
    inverse_homography = get_inverse_homography(src_ixt, tar_ixt, R, t, n_hat, a)  # (D, B, 3, 3)

    # Create grid from the ref frame
    tar_grid = create_meshgrid(H, W, device=depth.device).flip(-1)  # (H, W, 2), in xy ordering
    tar_grid = torch.cat((tar_grid, torch.ones_like(tar_grid[..., -1:])), -1)  # (H, W, 3)
    tar_grid = tar_grid[None, None].expand(D, B, -1, -1, -1).view(D, B, -1, 3)  # (D, B, H * W, 3)

    # Transform the target grid to the source frame
    src_grid = torch.matmul(inverse_homography, tar_grid.transpose(-2, -1)).transpose(-2, -1)  # (D, B, H * W, 3)
    src_grid = src_grid[..., :-1] / src_grid[..., -1:].clip(1e-6)  # (D, B, H * W, 2)
    src_grid = torch.cat([src_grid[..., :1, :] / W * 2 - 1,  # x
                          src_grid[..., 1:, :] / H * 2 - 1], dim=-2)  # y
    # (D, B, H * W, 2) -> (B, D, H * W, 2) -> (B, D, H, W, 2)
    src_grid = src_grid.permute(1, 0, 2, 3).reshape(B, D, H, W, 2)
    return src_grid


def build_plane_sweep_volume(depth: torch.Tensor, feat: torch.Tensor, mats: torch.Tensor, agg_method=0):
    """Construct a source image plane sweep volume in reference image coordinate.
    Args:
        depth (torch.Tensor): (B, D, H, W), sampled sweep volume depth of each plane, in reference camera coordinate.
        feat (torch.Tensor): (B, S, C, H, W), original rgb image or extract image feature.
        mats (torch.Tensor): (B, S, 3, 4), projection matrices from reference image to source image.
        agg_method (int) or (str): different aggregation method.
    Returns:
        plane_sweep_volume (torch.Tensor): (B, C, D, H, W), a plane sweep volume.
    """
    # The input feat may be original rgb image or extract image feature
    if feat.ndim < 5: feat = feat.unsqueeze(dim=-4)

    # Prepare shapes
    sh = feat.shape
    depth = depth[:, None].expand(-1, feat.shape[1], -1, -1, -1)  # (B, S, D, H, W), no memory allcoation
    depth = depth.reshape(-1, *depth.shape[-3:])  # (B * S, D, H, W), will allocate memory
    feat = feat.view(-1, *feat.shape[-3:])  # (B * S, C, H, W)
    mats = mats.view(-1, 3, 4)  # (B * S, 3, 4)

    BS, D, Ht, Wt = depth.shape
    BS, C, Hs, Ws = feat.shape

    R = mats[..., :3]  # (B * S, 3, 3)
    T = mats[..., 3:]  # (B * S, 3, 1)

    # Create grid from the ref frame
    ref_grid = create_meshgrid(Ht, Wt, device=depth.device).flip(-1)  # (H, W, 2), in xy ordering
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[..., -1:])), -1)  # (H, W, 3)
    ref_grid = ref_grid[None].expand(BS, -1, -1, -1).permute(0, 3, 1, 2)  # (B * S, 3, H, W)

    # Warp grid to source views
    src_grid = (R @ ref_grid.reshape(BS, 3, -1))  # (B * S, 3, H * W), apply rotation
    src_grid = src_grid[..., None, :].expand(-1, -1, D, -1).reshape(BS, 3, D * Ht * Wt)  # (B * S, 3, D * H * W), allocate memory
    src_grid = src_grid + T / depth.reshape(BS, 1, -1)  # (B * S, 3, D * H * W)
    src_grid = src_grid[..., :-1, :] / src_grid[..., -1:, :].clip(1e-6)  # (B * S, 2, D * H * W), divide by depth
    src_grid = torch.cat([src_grid[..., :1, :] / Ws * 2 - 1,  # x
                          src_grid[..., 1:, :] / Hs * 2 - 1], dim=-2)  # y
    src_grid = src_grid.permute(0, 2, 1)  # (B * S, D * H * W, 2)
    src_grid = src_grid.view(BS, D, Ht * Wt, 2)  # (B * S, D, H * W, 2)

    # Actual grid sample
    plane_sweep_vlume = F.grid_sample(feat, src_grid, padding_mode='border')  # (B * S, C, D, H * W)

    # Aggregate feature volume
    B, S = sh[:2]  # batch and all source images
    plane_sweep_vlume = plane_sweep_vlume.view(B, S, C, D, Ht, Wt)  # (B, S, C, D, H, W)
    if agg_method == 'var': plane_sweep_vlume = plane_sweep_vlume.var(dim=1)  # store variance as feature
    elif agg_method == 'sum': plane_sweep_vlume = plane_sweep_vlume.sum(dim=1)  # store sum as feature
    elif agg_method == 'mean': plane_sweep_vlume = plane_sweep_vlume.mean(dim=1)  # store mean as feature
    elif agg_method == 'median': plane_sweep_vlume = plane_sweep_vlume.median(dim=1)[0]  # store median as feature
    elif isinstance(agg_method, int): plane_sweep_vlume = plane_sweep_vlume[:, agg_method]  # particular index
    else: raise NotImplementedError

    return plane_sweep_vlume


def render_debug_plane_sweep_volume(t, z_vals, psv, batch):
    # Compute xyz coordinates of each ray sample
    B = batch.meta.H.shape[0]
    H, W, K, R, T = batch.meta.H[0].item(), batch.meta.W[0].item(), batch.ref_ixt, batch.ref_ext[:, :3, :3], batch.ref_ext[:, :3, 3:]
    ray_o, ray_d = get_rays(H, W, K, R, T)  # maybe without normalization
    ray_o, ray_d = ray_o.view(B, H * W, 3), ray_d.view(B, H * W, 3)  # (B, H * W, 3), (B, H * W, 3)
    xyz, dir, t, dist = ray2xyz(ray_o, ray_d, t, z_vals)  # (B, D * P, 3) == (B, D * H * W, 3)

    # (B, 3, D, H, W) -> (B, H, W, D, 3) -> (B, D * H * W, 3)
    psv = psv.permute(0, 3, 4, 2, 1).reshape(B, -1, 3)
    export_pts(xyz, color=psv, filename='psv.ply')
    __import__('easyvolcap.utils.console_utils', fromlist=['debugger']).debugger()


class LayerNorm(nn.Module):
    def __init__(self, out_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()

        self.out_channels = out_channels
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(out_channels).uniform_())
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        shape = [-1] + [1] * (x.ndim - 1)
        mean = x.contiguous().view(x.size(0), -1).mean(1).view(*shape)
        std = x.contiguous().view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.ndim - 2)
            y = self.weight.view(*shape) * y + self.bias.view(*shape)
        return y


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=None,
                 dilation=1, groups=1, bias=None, transpose=False,
                 norm_act=nn.BatchNorm2d, act_cls=nn.ReLU):
        super(ConvLayer, self).__init__()
        layers = []

        if pad is None: pad = (kernel_size - 1) // 2
        if transpose: layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                     stride=stride, padding=pad, groups=groups,
                                                     bias=bias, dilation=dilation))
        else: layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=pad, groups=groups,
                                    bias=bias, dilation=dilation))
        if norm_act is not None: layers.append(norm_act(out_channels))
        if act_cls is not None: layers.append(act_cls())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@REGRESSORS.register_module()
class StereoMagnificationNet(nn.Module):
    def __init__(self, n_planes, hidden_channels=64):
        super(StereoMagnificationNet, self).__init__()

        self.in_channels = 3 + n_planes * 3  # one reference image and n_planes source images
        self.out_channels = 3 + n_planes * 2  # one background image, N alpha, and N blend weights
        
        self.conv1_1 = ConvLayer(self.in_channels, hidden_channels,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 64, H, W)
        self.conv1_2 = ConvLayer(hidden_channels, hidden_channels*2,
                                 kernel_size=3, stride=2,
                                 norm_act=nn.InstanceNorm2d)  # (B, 128, H/2, W/2)

        self.conv2_1 = ConvLayer(hidden_channels*2, hidden_channels*2,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 128, H/2, W/2)
        self.conv2_2 = ConvLayer(hidden_channels*2, hidden_channels*4,
                                 kernel_size=3, stride=2,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)

        self.conv3_1 = ConvLayer(hidden_channels*4, hidden_channels*4,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)
        self.conv3_2 = ConvLayer(hidden_channels*4, hidden_channels*4,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)
        self.conv3_3 = ConvLayer(hidden_channels*4, hidden_channels*8,
                                 kernel_size=3, stride=2,
                                 norm_act=nn.InstanceNorm2d)  # (B, 512, H/8, W/8)

        self.conv4_1 = ConvLayer(hidden_channels*8, hidden_channels*8,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 512, H/8, W/8)
        self.conv4_2 = ConvLayer(hidden_channels*8, hidden_channels*8,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 512, H/8, W/8)
        self.conv4_3 = ConvLayer(hidden_channels*8, hidden_channels*8,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 512, H/8, W/8)

        self.conv5_1 = ConvLayer(hidden_channels*16, hidden_channels*4,
                                 kernel_size=4, stride=2, pad=1, transpose=True,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)
        self.conv5_2 = ConvLayer(hidden_channels*4, hidden_channels*4,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)
        self.conv5_3 = ConvLayer(hidden_channels*4, hidden_channels*4,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 256, H/4, W/4)

        self.conv6_1 = ConvLayer(hidden_channels*8, hidden_channels*2,
                                 kernel_size=4, stride=2, pad=1, transpose=True,
                                 norm_act=nn.InstanceNorm2d)  # (B, 128, H/2, W/2)
        self.conv6_2 = ConvLayer(hidden_channels*2, hidden_channels*2,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 128, H/2, W/2)

        self.conv7_1 = ConvLayer(hidden_channels*4, hidden_channels,
                                 kernel_size=4, stride=2, pad=1, transpose=True,
                                 norm_act=nn.InstanceNorm2d)  # (B, 64, H, W)
        self.conv7_2 = ConvLayer(hidden_channels, hidden_channels,
                                 kernel_size=3, stride=1,
                                 norm_act=nn.InstanceNorm2d)  # (B, 64, H, W)

        self.conv8_1 = ConvLayer(hidden_channels, self.out_channels,
                                 kernel_size=1, stride=1, bias=True,
                                 norm_act=None, act_cls=nn.Tanh)  # (B, 3+N*2, H, W)

        self.size_pad = 8  # input size should be divisible by 8

    def forward(self, x):
        x = self.conv1_1(x)
        out_conv1_2 = self.conv1_2(x)

        x = self.conv2_1(out_conv1_2)
        out_conv2_2 = self.conv2_2(x)

        x = self.conv3_1(out_conv2_2)
        x = self.conv3_2(x)
        out_conv3_3 = self.conv3_3(x)

        x = self.conv4_1(out_conv3_3)
        x = self.conv4_2(x)
        out_conv4_3 = self.conv4_3(x)

        # Add skip connections
        x = torch.cat([out_conv4_3, out_conv3_3], dim=1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        out_conv5_3 = self.conv5_3(x)

        # Add skip connections
        x = torch.cat([out_conv5_3, out_conv2_2], dim=1)

        x = self.conv6_1(x)
        out_conv6_2 = self.conv6_2(x)

        # Add skip connections
        x = torch.cat([out_conv6_2, out_conv1_2], dim=1)

        x = self.conv7_1(x)
        x = self.conv7_2(x)

        x = self.conv8_1(x)
        return x


def raw2mpi(raw, batch):
    """Convert raw stereo magnification network output to MPI representation.
    Args:
        raw (torch.Tensor): (B, 2 * D + 3, H, W), raw network output, which contains
                            1 background image, D alpha images and D blend weights.
        batch (dotdict): batch information, which contains the 'ref_inp' we need.
    Returns:
        mpi (torch.Tensor): (B, 3 * D, H, W), converted MPI representation.
    """
    # Nasty shape manipulation
    B, C, H, W = raw.shape  # (B, 2 * D + 3, H, W)
    D = (C - 3) // 2  # number of planes

    # Split raw into background, alpha and blend weights
    raw = raw.permute(0, 2, 3, 1)  # (B, H, W, 2 * D + 3)
    weights, alpha, bg, = raw.split([D, D, 3], dim=-1)  # (B, H, W, D), (B, H, W, D), (B, H, W, 3)
    # Rescale weights and alpha to range [0, 1]
    weights = (weights + 1.) / 2.  # (B, H, W, D)
    alpha = (alpha + 1.) / 2.  # (B, H, W, D)

    # Fetch reference image from batch
    fg = batch.ref_inp.permute(0, 2, 3, 1)  # (B, H, W, 3)

    # Compute MPI
    mpi_colors = weights[..., None] * fg[..., None, :] + (1 - weights[..., None]) * bg[..., None, :]  # (B, H, W, D, 3)
    mpi_alphas = alpha[..., None]  # (B, H, W, D, 1)
    mpi_rgba = torch.cat([mpi_colors, mpi_alphas], dim=-1)  # (B, H, W, D, 4)

    return mpi_rgba
