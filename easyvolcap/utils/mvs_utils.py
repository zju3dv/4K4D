import torch

from torch import nn
from typing import Union, List
from torch.nn import functional as F

from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.data_utils import export_pts, export_pcd, export_mesh
from easyvolcap.utils.enerf_utils import ConvBnReLU, ConvBnReLU3D, weights_init
from easyvolcap.utils.image_utils import interpolate_image
from easyvolcap.utils.nerf_utils import render_weights, raw2alpha
from easyvolcap.utils.math_utils import normalize, affine_inverse


def compute_scale(Hc: int, Wc: int,
                  Ho: int, Wo: int,
                  scale: float, round: int = 1,
                  ret: bool = False):
    """ Perform and compute the scaling factor for the image
    Args:
        Hc, Wc: (int, int), current image size, maybe rounded before for painless convolution and skip connection
        Ho, Wo: (int, int), original image size, may not equal to current image size
        scale: (float), scaling factor
        round: (int), round up the scaled image size to the nearest multiple of round
        ret: (bool), return the new scaling factor with respect to the original image size if True
    Returns:
        Ht, Wt: (int, int), scaled image size
        real_scale: (torch.Tensor), (2, 1), new scaling factor with respect to the original image size
    """
    # Compute the rescaled image size and round it up to the nearest multiple of round, default to 1
    Ht, Wt = int(Hc * scale), int(Wc * scale)
    Ht, Wt = int(np.ceil(Ht / round)) * round, int(np.ceil(Wt / round)) * round

    # Compute actual scaling and return if specified
    real_scale = torch.as_tensor([Wt / Wo, Ht / Ho])[..., None]  # (2, 1)

    if ret: return Ht, Wt, real_scale
    else: return Ht, Wt


def extremize_depth_map(near: torch.Tensor, far: torch.Tensor):
    """ Extremize the depth range to get the largest possible range
    Args:
        near: (torch.Tensor), (B, P, 1), near depth range for each pixel, may differ
        far: (torch.Tensor), (B, P, 1), far depth range for each pixel, may differ
    Returns:
        near_plane: (torch.Tensor), (B, 1, 1, 1), a uniform nearest depth plane for all pixels
        far_plane: (torch.Tensor), (B, 1, 1, 1), a uniform farthest depth plane for all pixels
    """
    # (batch_size, depth_hypotheses, height, width)
    near_plane = near.min(dim=-2)[0][..., None, None, :]  # (B, P, 1) -> (B, 1, 1, 1)
    far_plane = far.max(dim=-2)[0][..., None, None, :]  # (B, P, 1) -> (B, 1, 1, 1)
    return near_plane, far_plane


def upsample_depth_map(near: torch.Tensor, far: torch.Tensor,
                       Ht: int, Wt: int,
                       align_corners: bool = False):
    """ Upsample the previous depth range to guide depth estimation of current level
    Args:
        near: (torch.Tensor), (B, 1, Hc, Wc), low resolution near depth maps of previous level
        far: (torch.Tensor), (B, 1, Hc, Wc), low resolution far depth maps of previous level
        Ht, Wt: (int, int), the image height and width of current level
    Returns:
        near_plane: (torch.Tensor), (B, 1, Ht, Wt), upsampled nearest depth plane for current level
        far_plane: (torch.Tensor), (B, 1, Ht, Wt), upsampled farthest depth plane for current level
    """
    # Volume built around the surface
    near_plane = interpolate_image(near, align_corners=align_corners, size=(Ht, Wt))  # ? INTERP, (B, 1, Ht, Wt)
    far_plane = interpolate_image(far, align_corners=align_corners, size=(Ht, Wt))  # ? INTERP, (B, 1, Ht, Wt)
    return near_plane, far_plane


def topk_selection(depth_hypo: torch.Tensor, depth_prob: torch.Tensor, k: int):
    """ Sample next level depth hypotheses between top k intervals
    Args:
        depth_hypo: (torch.Tensor), (B, D, Hc, Wc), depth hypotheses of current level
        depth_prob: (torch.Tensor), (B, D, Hc, Wc), probability of each depth hypothsis
        k: (int), number of depth hypotheses needs to sample for next level
    Returns:
        multi_hypo: (torch.Tensor), (B, 2*k, Ht, Wt), sampled depth hypotheses for next level
    """
    # Interval computation, for later sample interval subdivision
    hypo_interval = depth_hypo[:, 1:] - depth_hypo[:, :-1]  # (B, D-1, Hc, Wc)
    hypo_interval = torch.cat([hypo_interval, hypo_interval[:, -1:]], dim=1)  # (B, D, Hc, Wc)

    # Top k selection
    _, selected_inds = torch.topk(depth_prob, k=k, dim=1)  # (B, Hc, Wc)
    selected_hypo = torch.gather(depth_hypo, dim=1, index=selected_inds)  # (B, k, Hc, Wc)
    selected_intl = torch.gather(hypo_interval, dim=1, index=selected_inds)  # (B, k, Hc, Wc)

    # subdivide depth interval and sample new depth hypotheses
    upper_hypo = selected_hypo + selected_intl / 2  # (B, k, Hc, Wc)
    lower_hypo = selected_hypo + selected_intl / 2  # (B, k, Hc, Wc)
    multi_hypo = torch.cat([lower_hypo, upper_hypo], dim=1)  # (B, 2*k, Hc, Wc)

    return multi_hypo


def mixtured_laplacian_depth_regression(depth_repr: torch.Tensor, eps: float = 1e-2):
    """ Perform depth regression from bimodal depth distribution, namely choosing the mode with
        the highest density value as the final prediction. The input of this function is the raw
        parameters of the mixtured Laplacian distribution, namely (μ1、μ2、λ1、λ2、c).
    Args:
        depth_repr: (torch.Tensor), (B, 5, H, W), bimodal depth distribution
        eps: (float), a small value to avoid numerical instability
    Returns:
        depth: (torch.Tensor), (B, H, W), estimated depth map
        std: (torch.Tensor), (B, H, W), estimated standard deviation, namely depth range
    """
    # Simplify notation
    activation = nn.Sigmoid()

    # Raw network output to bimodal distribution parameters
    mu0 = activation(depth_repr[:, 0, :])  # (B, H, W)
    mu1 = activation(depth_repr[:, 1, :])  # (B, H, W)
    sigma0 = torch.clamp(activation(depth_repr[:, 2, :]), eps, 1.0)  # (B, H, W)
    sigma1 = torch.clamp(activation(depth_repr[:, 3, :]), eps, 1.0)  # (B, H, W)
    pi0 = activation(depth_repr[:, 4, :])  # (B, H, W)
    pi1 = 1. - pi0  # (B, H, W)

    # Mode with the highest density value as final prediction
    mask = (pi0 / sigma0 > pi1 / sigma1).float()  # (B, H, W)

    # Compute the depth and standard deviation of the chosen mode
    depth = mu0 * mask + mu1 * (1. - mask)  # (B, H, W)
    std = sigma0 ** 2 * mask + sigma1 ** 2 * (1. - mask)  # (B, H, W)
    return depth, std


def unimodal_depth_regression(density: torch.Tensor, depth_hypo: torch.Tensor, is_debug: bool = False,
                              save_dir: str = None, level: int = 0):
    """ Perform depth regression from unimodal depth distribution, namely compute the expectation
        and standard deviation of the unimodal distribution, the difference of this function and its
        ancestor in `enerf_utils.py: depth_regression()` is the additional `Softplus()` activation
        before `softmax()` operation.
    Args:
        density: (torch.Tensor), (B, D, H, W), raw density of each pixel and each depth
        depth_hypo: (torch.Tensor), (B, D, H, W), depth planes or depth of samples
    Returns:
        depth: (torch.Tensor), (B, H, W), expectation of the unimodal distribution, namely the surface depth
        std: (torch.Tensor), (B, H, W), standard deviation of the unimodal distribution, possible range
    """
    # Use `softplus()` activation first, TODO: determine whether it is useful
    depth_prob = F.softplus(density)  # (B, D, H, W)

    # # Do not use `softplus()` as additional activation
    # depth_prob = density

    prob_volume = depth_prob.softmax(-3)  # (B, D, H, W)
    depth = (prob_volume * depth_hypo).sum(-3)  # (B, H, W)
    std = (prob_volume * (depth_hypo - depth.unsqueeze(-3))**2).sum(-3).clip(1e-10).sqrt()  # (B, H, W)

    # TODO: debug, save some useful middle output for visualization
    if is_debug:
        np.save(f'{save_dir}/depth_hypo_{level}.npy', depth_hypo[0].permute(1, 2, 0).detach().cpu().numpy())
        np.save(f'{save_dir}/raw_output_{level}.npy', density[0].permute(1, 2, 0).detach().cpu().numpy())
        np.save(f'{save_dir}/depth_prob_{level}.npy', prob_volume[0].permute(1, 2, 0).detach().cpu().numpy())

    return depth, std


def bimodal_depth_regression(density: torch.Tensor, z_vals: torch.Tensor, dist_default: float = 1.0,
                             is_debug: bool = False, save_dir: str = None, level: int = 0):
    """ Perform depth regression from bimodal depth distribution, there is no explicit mixtured
        Laplacian distribution here, otherweise, we consider the input `depth_prob` as density,
        like NeRF, we perform the diffrentiable volume rendering forward and backward to constrain
        the output to be bimodal. (this version sounds more reasonable than the version above?)
    Args:
        density: (torch.Tensor), (B, D, H, W), volume density of each pixel in every depth plane
        z_vals: (torch.Tensor), (B, D, H, W), depth planes
        dist_default: (float), default distance between depth planes
    Returns:
        mode1: (torch.Tensor), (B, H, W), the estimated first mode of the bimodal depth distribution
        mode2: (torch.Tensor), (B, H, W), the estimated second mode of the bimodal depth distribution
    """
    # Deal with nasty shapes
    B, D, H, W = density.shape
    density = density.reshape(B, D, -1).permute(0, 2, 1)  # (B, P, D)
    z_vals = z_vals.reshape(B, D, -1).permute(0, 2, 1)  # (B, P, D)

    # Compute the distance (distance for importance sampling only) between depth planes
    if dist_default < 0:
        dist = z_vals[..., 1:] - z_vals[..., :-1]  # (B, P, D)
        dist = torch.cat([dist, dist[..., -1:]], dim=-1)  # (B, P, D)
    else: dist = torch.full_like(density, dist_default)
    # Prepare for volume rendering: density * volume -> alpha (occupancy)
    occ = 1. - torch.exp(-F.softplus(density) * dist)  # (B, P, D)

    # Perform forward and backward volume rendering to constrain the output to be bimodal
    fore_weights = render_weights(occ[..., None])  # (B, P, D)
    back_weights = render_weights(occ.flip(dims=[-1,])[..., None])  # (B, P, D)

    # Normalize the two weights
    fore_weights, back_weights = normalize(fore_weights), normalize(back_weights)  # (B, P, D), (B, P, D)
    mode1 = (fore_weights * z_vals).sum(dim=-1).reshape(B, H, W)  # (B, H, W)
    mode2 = (back_weights * z_vals.flip(dims=[-1,])).sum(dim=-1).reshape(B, H, W)  # (B, H, W)

    # TODO: debug, save some useful middle output for visualization
    if is_debug:
        np.save(f'{save_dir}/depth_hypo_{level}.npy', z_vals[0].reshape(H, W, D).detach().cpu().numpy())
        np.save(f'{save_dir}/raw_output_{level}.npy', density[0].reshape(H, W, D).detach().cpu().numpy())
        np.save(f'{save_dir}/proc_alpha_{level}.npy', occ[0].reshape(H, W, D).detach().cpu().numpy())
        np.save(f'{save_dir}/fore_weights_{level}.npy', fore_weights[0].reshape(H, W, D).detach().cpu().numpy())
        np.save(f'{save_dir}/back_weights_{level}.npy', back_weights[0].reshape(H, W, D).detach().cpu().numpy())

    return mode1, mode2


def bimodal_depth_curve_fn(whole: torch.Tensor, depth: torch.Tensor):
    p = 0.01  # 1% high and 1% lows
    n = int(whole.numel() * p)
    near = whole.ravel().topk(n, largest=False)[0].max()  # a simple version of percentile
    far = whole.ravel().topk(n, largest=True)[0].min()  # a simple version of percentile
    depth = 1 - (depth - near) / (far - near)
    depth = depth.clip(0, 1)
    depth = depth.expand(depth.shape[:-1] + (3,))
    return depth


def sample_feature_volume(s_vals: torch.Tensor, uv: torch.Tensor, feat_vol: torch.Tensor):  # scale are in cpu to avoid sync
    """ Sample feature from feature volume, the difference between this function and the one in
        `enerf_utils.py` is this function does not assume a full resolution feature volume constrcution,
        it only builds upon those sampled pixel coordinates.
    Args:
        s_vals: (torch.Tensor), (B, P, N), s_vals for sampled points
        uv: (torch.Tensor), (B, P, N, 2), processed 2d image coordinate of sampled points, make sure it has expanded (to fit with nerfacc)
        feat_vol: (torch.Tensor), (B, C, D, Ht, Wt), constructed image feature volume of target scale
    Returns:
        vox_feat: (torch.Tensor), (B, P, C), sampled feature from feature volume
    """
    # Deal with nasty shapes
    B, P, N = s_vals.shape

    # Feature volume is w.r.t the target view size
    d = (s_vals * 2 - 1)[..., None]  # (B, P, N, 1), normalize to -1, 1
    uvd = torch.cat([uv, d], dim=-1)[:, None]  # (B, P, N, 3) -> (B, 1, P, N, 3)

    # Grid sample
    vox_feat = F.grid_sample(feat_vol, uvd, padding_mode='border')[:, :, 0]  # (B, C, 1, P, N) -> (B, C, P, N)
    vox_feat = vox_feat.permute(0, 2, 3, 1).view(B, P * N, -1)  # (B, P, N, C) -> (B, P*N, C)
    return vox_feat


def sample_feature_image(xyz: torch.Tensor, src_feat: torch.Tensor, src_inps: torch.Tensor,
                         tar_ext: torch.Tensor, src_exts: torch.Tensor,
                         src_scale: Union[float, torch.Tensor], src_ixts: torch.Tensor,
                         correct_pix: bool = False, compute_msk: bool = False):
    """ Sample feature from image, the difference between this function and the one in
        `enerf_utils.py` is this function does not require a pre-aligned and concatenated
        [src_feat, src_inps], which means the input `src_inps` is in full resolution,
        but `src_feat` may be in low resolution, say H//4, W//4.
    Args:
        xyz: (torch.Tensor), (B, P, 3), 3d world coordinates of sampled points
        src_feat: (torch.Tensor), (B, S, C, Hs, Ws), image feature of source views
        src_inps: (torch.Tensor), (B, S, 3, Hp, Wp), image of source views
        tar_ext: (torch.Tensor), (B, 4, 4), target camera extrinsic
        src_exts: (torch.Tensor), (B, S, 4, 4), source camera extrinsics
        src_scale: (Union[float, torch.Tensor]), (2, 1), scaling factor of source views
        src_ixts: (torch.Tensor), (B, S, 3, 3), source camera intrinsics
    Returns:
        img_feats: (torch.Tensor), (B, S, P, C), feature for all sampled points
    """
    # Deal with nasty shapes
    B, S, C, Hs, Ws = src_feat.shape
    _, _, _, Hp, Wp = src_inps.shape
    _, P, _ = xyz.shape

    # compensate the shift caused by padding `src_inps` for painless up convolution and skip connections
    src_ixts = src_ixts.clone()  # (B, S, 3, 3)
    src_ixts[..., :2, :] *= src_scale  # (B, S, 3, 3)

    xyz1 = torch.cat([xyz, torch.ones_like(xyz[..., -1:])], dim=-1)  # (B, P, 4), homogeneous coordinates
    xyz1 = (xyz1[..., None, :, :] @ src_exts.mT)  # (B, S, P, 4)
    xyzs = xyz1[..., :3] @ src_ixts.mT  # (B, S, P, 3) @ (B, S, 3, 3) -> (B, S, P, 3)
    xy = xyzs[..., :-1] / torch.clamp_min(xyzs[..., -1:], 1e-6)  # (B, S, P, 2)
    x, y = xy.chunk(2, dim=-1)  # (B, S, P, 1), (B, S, P, 1)
    if correct_pix: xy = torch.cat([x / Wp * 2 - 1, y / Hp * 2 - 1], dim=-1)  # (B, S, P, 2)
    else: xy = torch.cat([x / (Wp - 1) * 2 - 1, y / (Hp - 1) * 2 - 1], dim=-1)  # (B, S, P, 2)

    # Actual sampling of the image features and rgb colors, (BS, C, 1, P) -> (B, S, C, P) -> (B, S, P, C)
    pts_inps = F.grid_sample(src_inps.view(-1, 3, Hp, Wp), xy.view(-1, 1, P, 2), padding_mode='border', align_corners=not correct_pix).view(B, S, 3, P).permute(0, 1, 3, 2)
    pts_feat = F.grid_sample(src_feat.view(-1, C, Hs, Ws), xy.view(-1, 1, P, 2), padding_mode='border', align_corners=not correct_pix).view(B, S, C, P).permute(0, 1, 3, 2)

    # Computing directional features
    # ENeRF firstly computes direction features and then aggregates them
    # For xyz, use aggregated feature
    # For dir, cat with directional feature
    tar_cam = affine_inverse(tar_ext)[..., :3, -1:].mT  # (B, 1, 3)
    src_cams = affine_inverse(src_exts)[..., :3, -1:].mT  # (B, S, 1, 3)
    tar_ray = normalize(xyz - tar_cam)  # (B, P, 3)
    src_rays = normalize(xyz[:, None] - src_cams)  # (B, S, P, 3)
    ray_diff_dir = normalize(tar_ray[:, None] - src_rays)  # (B, S, P, 3)
    ray_diff_dot = (tar_ray[:, None] * src_rays).sum(dim=-1, keepdim=True)  # (B, S, P, 1)

    # https://github.com/haotongl/htcode/blob/main/lib/networks/nerf/encoders/imgfeat_ibrnet.py
    # Additional mask feature, may good for non-overlapping area
    if compute_msk: pts_mask = ((xy[..., 0] >= -1) & (xy[..., 0] <= 1) & (xy[..., 1] >= -1) & (xy[..., 1] <= 1) & (xyzs[..., 2] >= 1e-6))[..., None].float()  # (B, S, P, 1)
    else: pts_mask = torch.empty_like(ray_diff_dot[..., :0])  # (B, S, P, 0)

    # Order matters, see how `ImageBasedRegressor` in `image_based_regressor.py` splits rgb from it
    img_feats = torch.cat([pts_mask, pts_feat, pts_inps, ray_diff_dir, ray_diff_dot], dim=-1)
    return img_feats  # (B, S, P, C)


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                     groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_channels, out_channels, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                     bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_act=nn.BatchNorm2d):

        super(BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if groups != 1 or base_width != 64: raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_act(out_channels, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_act(out_channels, track_running_stats=False, affine=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(self.kernel_size - 1) // 2, padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(out_channels, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(in_channels, out_channels, kernel_size, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


@REGRESSORS.register_module()
class FeaturenetIbrnet(nn.Module):
    """ This a configurable 2D image featurenet, users can define the encoder type, coarse output channel,
        fine level output channel (and whether to output a fine level), it is more flexible than the original
        implementation in `enerf_utils.py`.
    """

    def __init__(self, coarse_dims=32, coarse_only=False, fine_dims=32,
                 encoder='resnet34', norm_act=nn.InstanceNorm2d):

        super(FeaturenetIbrnet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Unsupport encoder type"
        if encoder in ['resnet18', 'resnet34']: filters = [64, 128, 256, 512]
        else: filters = [256, 512, 1024, 2048]

        # Configurations
        self.coarse_only = coarse_only
        self.coarse_dims = coarse_dims
        self.fine_dims = fine_dims if not self.coarse_only else 0
        self.out_dim = coarse_dims + fine_dims

        self.groups = 1
        self.dilation = 1
        self.base_width = 64
        self.in_channels = 64
        self._norm_act = norm_act

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='reflect')
        self.bn1 = norm_act(self.in_channels, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        block = BasicBlock
        layers = [3, 4, 6, 3]
        replace_stride_with_dilation = [False, False, False]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        # Decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, self.out_dim, 3, 1)

        # Output
        self.out_conv = nn.Conv2d(self.out_dim, self.out_dim, 1, 1)

        self.out_dims = [self.coarse_dims, self.fine_dims]  # output dimensionality
        self.scales = [0.25, 0.25]
        self.size_pad = 8  # input size should be divisible by 8

    def _make_layer(self, block, out_channels, blocks, stride=1, dilate=False):
        norm_act = self._norm_act
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_act(out_channels * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_act))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_act=norm_act))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # For padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # In order to maintain the same API with the original feature extractor that enerf use,
        # we need to reshape the input tensor to (B*S, C, H, W) and reshape the output tensor
        sh = x.shape
        x = x.view(-1, *sh[-3:])  # (B*S, C, H, W) or (C, H, W)

        x = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)
        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)  # (B*S, C, H, W) or (C, H, W)
        x_out = x_out.view(sh[:-3] + x_out.shape[-3:])  # (B, S, C, H, W) or (C, H, W)
        if self.coarse_only: x_fine, x_coarse = None, x_out
        else: x_fine, x_coarse = x_out[..., :self.fine_dims, :, :], x_out[..., -self.coarse_dims:, :, :]

        return x_fine, x_coarse


def fused_mean_variance(x, weight):
    mean = torch.sum(x * weight, dim=-2, keepdim=True)
    var = torch.sum(weight * (x - mean) ** 2, dim=-2, keepdim=True)
    return mean, var


@REGRESSORS.register_module()
class FeatureAggIBRNet(nn.Module):
    def __init__(self, feat_ch, use_mvs_dens,
                 viewdir_agg=True,
                 anti_alias_pooling=True):

        super(FeatureAggIBRNet, self).__init__()
        self.feat_ch = feat_ch
        self.viewdir_agg = viewdir_agg
        self.anti_alias_pooling = anti_alias_pooling
        self.use_mvs_dens = use_mvs_dens
        self.act_func = nn.ELU(inplace=True)

        # https://github.com/haotongl/htcode/blob/b66768fcb2fca09b757c739288ba334cf432f4a9/lib/networks/nerf/fields/ibrnet_net.py#L157
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)

        # Layered ENeRF ignores viewdir during vanilla xyz embedding
        if self.viewdir_agg:
            self.view_fc = nn.Sequential(nn.Linear(4, 16), self.act_func, nn.Linear(16, feat_ch), self.act_func)
            self.view_fc.apply(weights_init)

        self.base_fc = nn.Sequential(nn.Linear(feat_ch * 3, 64), self.act_func, nn.Linear(64, 32), self.act_func)
        self.vis1_fc = nn.Sequential(nn.Linear(32, 32), self.act_func, nn.Linear(32, 33), self.act_func)
        self.vis2_fc = nn.Sequential(nn.Linear(32, 32), self.act_func, nn.Linear(32, 1), self.act_func)

        if not self.use_mvs_dens:
            self.geo_fc = nn.Sequential(nn.Linear(32 * 2 + 1, 64), self.act_func, nn.Linear(64, 16), self.act_func)
            self.out_geo_fc = nn.Sequential(nn.Linear(16, 16), self.act_func, nn.Linear(16, 1), nn.ReLU())

        self.base_fc.apply(weights_init)
        self.vis1_fc.apply(weights_init)
        self.vis2_fc.apply(weights_init)
        if not self.use_mvs_dens: self.geo_fc.apply(weights_init)

        self.out_dim = 37

    def forward(self, img_feat_rgb_dir: torch.Tensor):
        # Prepare shapes, [1, Cf, 3, 3, 1] where `Cf + 3 = self.feat_ch`, which corresponds to
        # `img_feat_rgb_dir` = [`pts_mask`, `pts_feat`, `pts_inps`, `ray_diff_dir`, `ray_diff_dot`]
        img_feat_rgb_dir = img_feat_rgb_dir.permute(0, 2, 1, 3)  # (B, S, P, C) -> (B, P, S, C)
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]

        if self.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., 1 + self.feat_ch:])  # (B, P, S, Cf)
            img_feat_rgb = img_feat_rgb_dir[..., 1:1 + self.feat_ch] + view_feat  # (B, P, S, Cf)
        else:
            img_feat_rgb = img_feat_rgb_dir[..., 1:1 + self.feat_ch]  # (B, P, S, Cf)

        mask = img_feat_rgb_dir[..., :1]  # (B, P, S, 1)
        if self.anti_alias_pooling:
            exp_dot_prod = torch.exp(torch.abs(self.s) * (img_feat_rgb_dir[..., -1:] - 1))  # (B, P, S, 1)
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=-2, keepdim=True)[0]) * mask  # (B, P, S, 1)
            weight = weight / (torch.sum(weight, dim=-2, keepdim=True) + 1e-8)  # (B, P, S, 1)
        else:
            weight = mask / (torch.sum(mask, dim=-2, keepdim=True) + 1e-8)  # (B, P, S, 1)

        # Compute mean and variance of `img_feat_rgb` across different views for each point
        avg_feat, var_feat = fused_mean_variance(img_feat_rgb, weight)  # (B, P, 1, Cf), (B, P, 1, Cf)
        img_feat = torch.cat([torch.cat([avg_feat, var_feat], dim=-1).expand(-1, -1, S, -1), img_feat_rgb], dim=-1)  # (B, P, S, 3*Cf)
        img_feat = self.base_fc(img_feat)  # (B, P, S, 32)

        # Compute sourve view visibilities, from Neuray Rays?
        vis_feat = self.vis1_fc(img_feat * weight)  # (B, P, S, 32)
        res_feat, visibility = torch.split(vis_feat, [vis_feat.shape[-1] - 1, 1], dim=-1)  # (B, P, S, 32), (B, P, S, 1)
        img_feat = img_feat + res_feat  # (B, P, S, 32)
        visibility = self.vis2_fc(img_feat * (F.sigmoid(visibility) * mask)) * mask  # (B, P, S, 1)

        # Concatenate appearance feature, [`img_feat`, `visibility`, `ray_diff_dir`, `ray_diff_dot`]
        app_feat = torch.cat([img_feat, visibility, img_feat_rgb_dir[..., 1 + self.feat_ch:]], dim=-1)  # (B, P, S, 32+1+4)
        app_feat = app_feat.permute(0, 2, 1, 3)  # (B, P, S, 32+1+4) -> (B, S, P, 32+1+4)

        # Compute density here if `use_mvs_dens` is False, which means `self.geometry` in `VolumetricVideoNetwork`
        # is always `NoopRegressor`, its output consists of [`density`, `app_feat`] computed in `IbrEmbedder`
        if self.use_mvs_dens: density = None
        else:
            vis_weight = visibility / (torch.sum(visibility, dim=-2, keepdim=True) + 1e-8)  # (B, P, S, 1)
            avg_feat, var_feat = fused_mean_variance(img_feat, vis_weight)  # (B, P, 1, 32), (B, P, 1, 32)
            geo_feat = torch.cat([avg_feat.squeeze(-2), var_feat.squeeze(-2), vis_weight.mean(dim=-2)], dim=-1)  # (B, P, 32+32+1)
            geo_feat = self.geo_fc(geo_feat)  # (B, P, 16)
            density = self.out_geo_fc(geo_feat)  # (B, P, 1)
            # set the sigma of invalid point to zero
            density = density.masked_fill(torch.sum(mask, dim=-2) < 1, 0.)

        return density, app_feat  # (B, P, 1), (B, S, P, 37)
