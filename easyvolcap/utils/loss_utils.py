import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import namedtuple

from easyvolcap.utils.prop_utils import searchsorted, matchup_channels

from enum import Enum, auto

class ElasticLossReduceType(Enum):
    WEIGHT = auto()
    MEDIAN = auto()


class ImgLossType(Enum):
    PERC = auto()  # lpips
    CHARB = auto()
    HUBER = auto()
    L1 = auto()
    L2 = auto()
    SSIM = auto()

# from mipnerf360


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)  # 129
    idx_lo, idx_hi = searchsorted(t1, t0)

    cy1_lo = torch.take_along_dim(cy1, idx_lo, dim=-1)  # 128
    cy1_hi = torch.take_along_dim(cy1, idx_hi, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]  # 127
    y0_inner = torch.where(idx_hi[..., :-1] <= idx_lo[..., 1:], cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
    return y0_inner, y0_outer

# from mipnerf360


def lossfun_outer(t: torch.Tensor, w: torch.Tensor, t_env: torch.Tensor, w_env: torch.Tensor, eps=torch.finfo(torch.float32).eps):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)
    """The proposal weight should be an upper envelope on the nerf weight."""
    _, w_outer = inner_outer(t, t_env, w_env)
    # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
    # more effective to pull w_outer up than it is to push w_inner down.
    # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
    return (w - w_outer).clip(0.).pow(2) / (w + eps)


def blur_stepfun(x, y, r):
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) -
          torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) *
                      torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    return xr, yr


def sorted_interp_quad(x, xp, fpdf, fcdf):
    """interp in quadratic"""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x, return_idx=False):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, x0_idx = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, x1_idx = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        if return_idx:
            return x0, x1, x0_idx, x1_idx
        return x0, x1

    fcdf0, fcdf1, fcdf0_idx, fcdf1_idx = find_interval(fcdf, return_idx=True)
    fpdf0 = fpdf.take_along_dim(fcdf0_idx, dim=-1)
    fpdf1 = fpdf.take_along_dim(fcdf1_idx, dim=-1)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fcdf0 + (x - xp0) * (fpdf0 + fpdf1 * offset + fpdf0 * (1 - offset)) / 2
    return ret


def lossfun_zip_outer(t, w, t_env, w_env, pulse_width, eps=1e-6):
    t, w = matchup_channels(t, w)
    t_env, w_env = matchup_channels(t_env, w_env)

    w_normalize = w / torch.clamp_min(t[..., 1:] - t[..., :-1], eps)

    t_, w_ = blur_stepfun(t, w_normalize, pulse_width)
    w_ = torch.clip(w_, min=0.)
    assert (w_ >= 0.0).all()

    # piecewise linear pdf to piecewise quadratic cdf
    area = 0.5 * (w_[..., 1:] + w_[..., :-1]) * (t_[..., 1:] - t_[..., :-1])

    cdf = torch.cat([torch.zeros_like(area[..., :1]), torch.cumsum(area, dim=-1)], dim=-1)

    # query piecewise quadratic interpolation
    cdf_interp = sorted_interp_quad(t_env, t_, w_, cdf)
    # difference between adjacent interpolated values
    w_s = torch.diff(cdf_interp, dim=-1)

    return ((w_s - w_env).clip(0.).pow(2) / (w_env + eps)).mean()


def lossfun_distortion(t: torch.Tensor, w: torch.Tensor):
    # accepts t.shape[-1] = w.shape[-1] + 1
    t, w = matchup_channels(t, w)
    """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
    # The loss incurred between all pairs of intervals.
    ut = (t[..., 1:] + t[..., :-1]) / 2  # 64
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])  # 64
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)

    # The loss incurred within each individual interval with itself.
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
    """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
    # Distortion when the intervals do not overlap.
    d_disjoint = torch.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

    # Distortion when the intervals overlap.
    d_overlap = (2 *
                 (torch.minimum(t0_hi, t1_hi)**3 - torch.maximum(t0_lo, t1_lo)**3) +
                 3 * (t1_hi * t0_hi * torch.abs(t1_hi - t0_hi) +
                      t1_lo * t0_lo * torch.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                      (t0_lo - t1_hi) + t1_lo * t0_hi *
                      (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

    # Are the two intervals not overlapping?
    are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

    return torch.where(are_disjoint, d_disjoint, d_overlap)


def anneal_loss_weight(weight: float, gamma: float, iter: int, mile: int):
    # exponentially anneal the loss weight
    return weight * gamma ** min(iter / mile, 1)


def gaussian_entropy_relighting4d(albedo_pred):
    albedo_entropy = 0
    for i in range(3):
        channel = albedo_pred[..., i]
        hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
        h = hist(channel)
        if h.sum() > 1e-6:
            h = h.div(h.sum()) + 1e-6
        else:
            h = torch.ones_like(h)
        albedo_entropy += torch.sum(-h * torch.log(h))
    return albedo_entropy


class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=sigma.device).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma)**2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=1)
        return x


def gaussian_entropy(x: torch.Tensor, *args, **kwargs):
    eps = 1e-6
    hps = 1e-9
    h = gaussian_histogram(x, *args, **kwargs)
    # h = (h / (h.sum(dim=0) + hps)).clip(eps)  # 3,
    # entropy = (-h * h.log()).sum(dim=0).sum(dim=0)  # per channel entropy summed
    entropy = 0
    for i in range(3):
        hi = h[..., i]
        if hi.sum() > eps:
            hi = hi / hi.sum() + eps
        else:
            hi = torch.ones_like(hi)
        entropy += torch.sum(-hi * torch.log(hi))
    return entropy


def gaussian_histogram(x: torch.Tensor, bins: int = 15, min: float = 0.0, max: float = 1.0):
    x = x.view(-1, x.shape[-1])  # N, 3
    sigma = x.var(dim=0)  # 3,
    delta = (max - min) / bins
    centers = min + delta * (torch.arange(bins, device=x.device, dtype=x.dtype) + 0.5)  # BIN
    x = x[None] - centers[:, None, None]  # BIN, N, 3
    x = (-0.5 * (x / sigma).pow(2)).exp() / (sigma * np.sqrt(np.pi * 2)) * delta  # BIN, N, 3
    x = x.sum(dim=1)
    return x  # BIN, 3


def reg_diff_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    return reg(x), weight


def reg_raw_crit(x: torch.Tensor, iter_step: int, max_weight: float = 1e-4, ann_iter: int = 100 * 500):
    weight = min(iter_step, ann_iter) * max_weight / ann_iter
    n_batch, n_pts_x2, D = x.shape
    n_pts = n_pts_x2 // 2
    length = x.norm(dim=-1, keepdim=True)  # length
    vector = x / (length + 1e-8)  # vector direction (normalized to unit sphere)
    # loss_length = mse(length[:, n_pts:, :], length[:, :n_pts, :])
    loss_vector = reg((vector[:, n_pts:, :] - vector[:, :n_pts, :]))
    # loss = loss_length + loss_vector
    loss = loss_vector
    return loss, weight


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        try:
            from torchvision.models import VGG19_Weights
            self.vgg_layers = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        except ImportError:
            self.vgg_layers = vgg.vgg19(pretrained=True).features

        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''

        self.layer_name_mapping = {'3': "relu1", '8': "relu2"}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name == '8':
                break
        LossOutput = namedtuple("LossOutput", ["relu1", "relu2"])
        return LossOutput(**output)


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

    def forward(self, x, target):
        x_feature = self.model(x[:, 0:3, :, :])
        target_feature = self.model(target[:, 0:3, :, :])

        feature_loss = (
            self.l1_loss(x_feature.relu1, target_feature.relu1) +
            self.l1_loss(x_feature.relu2, target_feature.relu2)) / 2.0

        l1_loss = self.l1_loss(x, target)
        l2_loss = self.mse_loss(x, target)

        loss = feature_loss + l1_loss + l2_loss

        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        import torchvision
        vgg16 = torchvision.models.vgg16(pretrained=True)
        blocks.append(vgg16.features[:4].eval())
        blocks.append(vgg16.features[4:9].eval())
        blocks.append(vgg16.features[9:16].eval())
        blocks.append(vgg16.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += F.l1_loss(gram_x, gram_y)
        return loss


def eikonal(x: torch.Tensor, th=1.0) -> torch.Tensor:
    return ((x.norm(dim=-1) - th)**2).mean()


def sdf_mask_crit(ret, batch):
    msk_sdf = ret['msk_sdf']
    msk_label = ret['msk_label']

    alpha = 50
    alpha_factor = 2
    alpha_milestones = [10000, 20000, 30000, 40000, 50000]
    for milestone in alpha_milestones:
        if batch['iter_step'] > milestone:
            alpha = alpha * alpha_factor

    msk_sdf = -alpha * msk_sdf
    mask_loss = F.binary_cross_entropy_with_logits(msk_sdf, msk_label) / alpha

    return mask_loss


def cross_entropy(x: torch.Tensor, y: torch.Tensor):
    # x: unormalized input logits
    # channel last cross entropy loss
    x = x.view(-1, x.shape[-1])  # N, C
    y = y.view(-1, y.shape[-1])  # N, C
    return F.cross_entropy(x, y)


def huber(x: torch.Tensor, y: torch.Tensor):
    return F.huber_loss(x, y, reduction='mean')


def smoothl1(x: torch.Tensor, y: torch.Tensor):
    return F.smooth_l1_loss(x, y)


def mse(x: torch.Tensor, y: torch.Tensor):
    return ((x.float() - y.float())**2).mean()


def dot(x: torch.Tensor, y: torch.Tensor):
    return (x * y).sum(dim=-1)


def l1(x: torch.Tensor, y: torch.Tensor):
    return l1_reg(x - y)


def l2(x: torch.Tensor, y: torch.Tensor):
    return l2_reg(x - y)


def l1_reg(x: torch.Tensor):
    return x.abs().sum(dim=-1).mean()


def l2_reg(x: torch.Tensor) -> torch.Tensor:
    return (x**2).sum(dim=-1).mean()


def bce_loss(x: torch.Tensor, y: torch.Tensor):
    return F.binary_cross_entropy(x, y)


def mIoU_loss(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the mean intersection of union loss over masked regions
    x, y: B, N, 1
    """
    I = (x * y).sum(-1).sum(-1)
    U = (x + y).sum(-1).sum(-1) - I
    mIoU = (I / U.detach()).mean()
    return 1 - mIoU


def reg(x: torch.Tensor) -> torch.Tensor:
    return x.norm(dim=-1).mean()


def thresh(x: torch.Tensor, a: torch.Tensor, eps: float = 1e-8):
    return 1 / (l2(x, a) + eps)


def elastic_crit(jac: torch.Tensor) -> torch.Tensor:
    """Compute the raw 'log_svals' type elastic energy, and
    remap it using the Geman-McClure type of robust loss.
    Args:
        jac (torch.Tensor): (B, N, 3, 3), the gradient of warpped xyz with respect to the original xyz
    Return:
        elastic_loss (torch.Tensor): (B, N), 
    """
    # !: CUDA IMPLEMENTATION OF SVD IS EXTREMELY SLOW
    # old_device = jac.device
    # jac = jac.cpu()
    # svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, and hence we cannot compute backward. Please use torch.svd(compute_uv=True)
    _, S, _ = torch.svd(jac, compute_uv=True)           # (B, N, 3)
    # S = S.to(old_device)
    log_svals = torch.log(torch.clamp(S, min=1e-6))     # (B, N, 3)
    sq_residual = torch.sum(log_svals**2, dim=-1)       # (B, N)
    # TODO: determine whether it is a good choice to compute the robust loss here
    elastic_loss = general_loss_with_squared_residual(sq_residual, alpha=-2.0, scale=0.03)
    return elastic_loss


def general_loss_with_squared_residual(squared_x, alpha, scale):
    r"""The general loss that takes a squared residual.
    This fuses the sqrt operation done to compute many residuals while preserving
    the square in the loss formulation.
    This implements the rho(x, \alpha, c) function described in "A General and
    Adaptive Robust Loss Function", Jonathan T. Barron,
    https://arxiv.org/abs/1701.03077.
    Args:
        squared_x: The residual for which the loss is being computed. x can have
        any shape, and alpha and scale will be broadcasted to match x's shape if
        necessary.
        alpha: The shape parameter of the loss (\alpha in the paper), where more
        negative values produce a loss with more robust behavior (outliers "cost"
        less), and more positive values produce a loss with less robust behavior
        (outliers are penalized more heavily). Alpha can be any value in
        [-infinity, infinity], but the gradient of the loss with respect to alpha
        is 0 at -infinity, infinity, 0, and 2. Varying alpha allows for smooth
        interpolation between several discrete robust losses:
            alpha=-Infinity: Welsch/Leclerc Loss.
            alpha=-2: Geman-McClure loss.
            alpha=0: Cauchy/Lortentzian loss.
            alpha=1: Charbonnier/pseudo-Huber loss.
            alpha=2: L2 loss.
        scale: The scale parameter of the loss. When |x| < scale, the loss is an
        L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        different shape according to alpha.
    Returns:
        The losses for each element of x, in the same shape as x.
    """
    # https://pytorch.org/docs/stable/type_info.html
    eps = torch.tensor(torch.finfo(torch.float32).eps)

    # convert the float to torch.tensor
    alpha = torch.tensor(alpha).to(squared_x.device)
    scale = torch.tensor(scale).to(squared_x.device)

    # This will be used repeatedly.
    squared_scaled_x = squared_x / (scale ** 2)

    # The loss when alpha == 2.
    loss_two = 0.5 * squared_scaled_x
    # The loss when alpha == 0.
    loss_zero = log1p_safe(0.5 * squared_scaled_x)
    # The loss when alpha == -infinity.
    loss_neginf = -torch.expm1(-0.5 * squared_scaled_x)
    # The loss when alpha == +infinity.
    loss_posinf = expm1_safe(0.5 * squared_scaled_x)

    # The loss when not in one of the above special cases.
    # Clamp |2-alpha| to be >= machine epsilon so that it's safe to divide by.
    beta_safe = torch.maximum(eps, torch.abs(alpha - 2.))
    # Clamp |alpha| to be >= machine epsilon so that it's safe to divide by.
    alpha_safe = torch.where(
        torch.greater_equal(alpha, torch.tensor(0.)), torch.ones_like(alpha),
        -torch.ones_like(alpha)) * torch.maximum(eps, torch.abs(alpha))
    loss_otherwise = (beta_safe / alpha_safe) * (
        torch.pow(squared_scaled_x / beta_safe + 1., 0.5 * alpha) - 1.)

    # Select which of the cases of the loss to return.
    loss = torch.where(
        alpha == -torch.inf, loss_neginf,
        torch.where(
            alpha == 0, loss_zero,
            torch.where(
                alpha == 2, loss_two,
                torch.where(alpha == torch.inf, loss_posinf, loss_otherwise))))

    return scale * loss


def log1p_safe(x):
    """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
    return torch.log1p(torch.minimum(x, torch.tensor(3e37)))


def expm1_safe(x):
    """The same as torch.expm1(x), but clamps the input to prevent NaNs."""
    return torch.expm1(torch.minimum(x, torch.tensor(87.5)))


def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_planes_tv(embedding):
    tv_loss = 0
    for emb in embedding:
        tv_loss += compute_plane_tv(emb)
    return tv_loss


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:] - t[..., :w - 1]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:] - first_difference[..., :w - 2]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def compute_time_planes_smooth(embedding):
    loss = 0.
    for emb in embedding:
        loss += compute_plane_smoothness(emb)
    return loss


def compute_ssim(x: torch.Tensor, y: torch.Tensor):
    from pytorch_msssim import ssim
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    return ssim(x, y, data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))
