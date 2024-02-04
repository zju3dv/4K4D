import torch
import numpy as np

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.loss_utils import mse as compute_mse

from skimage.metrics import structural_similarity as compare_ssim


def psnr(x: torch.Tensor, y: torch.Tensor):
    mse = compute_mse(x, y).mean()
    psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)
    return psnr.item()  # tensor to scalar


def ssim(xs: torch.Tensor, ys: torch.Tensor):
    return np.mean([
        compare_ssim(
            x.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            channel_axis=-1,
            data_range=2.0
        )
        for x, y in zip(xs, ys)
    ]).astype(float).item()


def lpips(x: torch.Tensor, y: torch.Tensor):
    # B, H, W, 3
    # B, H, W, 3
    if not hasattr(lpips, 'compute_lpips'):
        import lpips as lpips_module
        log('Initializing LPIPS network')
        lpips.compute_lpips = lpips_module.LPIPS(net='vgg', verbose=False).cuda()

    if x.ndim == 3: x = x.unsqueeze(0)
    if y.ndim == 3: y = y.unsqueeze(0)

    return lpips.compute_lpips(x.cuda().permute(0, 3, 1, 2) * 2 - 1, y.cuda().permute(0, 3, 1, 2) * 2 - 1).mean().item()
