# Default loss module (called supervisor)
import torch
import numpy as np
from torch import nn
from typing import Union

from easyvolcap.engine import SUPERVISORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.net_utils import VolumetricVideoModule
from easyvolcap.utils.loss_utils import l1, huber, compute_ssim, VGGPerceptualLoss, ImgLossType


@SUPERVISORS.register_module()
class VolumetricVideoSupervisor(VolumetricVideoModule):
    def __init__(self,
                 network: nn.Module,
                 img_loss_weight: float = 1.0,  # main reconstruction loss
                 img_loss_type: ImgLossType = ImgLossType.HUBER.name,  # chabonier loss for img_loss
                 perc_loss_weight: float = 0.0,  # smaller loss on perc
                 ssim_loss_weight: float = 0.0,  # 3dgs ssim loss
                 dtype: Union[str, torch.dtype] = torch.float,

                 **kwargs,
                 ):
        super().__init__(network)
        self.forward = self.supervise
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # Image reconstruction loss
        self.img_loss_weight = img_loss_weight
        self.img_loss_type = ImgLossType[img_loss_type]
        self.perc_loss_weight = perc_loss_weight
        self.ssim_loss_weight = ssim_loss_weight

    @property
    def perc_loss(self):
        return self.perc_loss_reference[0]

    def compute_image_loss(self, rgb_map: torch.Tensor, rgb_gt: torch.Tensor,
                           bg_color: torch.Tensor, msk_gt: torch.Tensor,
                           H: int, W: int,
                           type=ImgLossType.HUBER):
        rgb_gt = rgb_gt + bg_color * (1 - msk_gt)  # MARK: modifying gt for supervision

        # https://stackoverflow.com/questions/181530/styling-multi-line-conditions-in-if-statements
        resd_sq = (rgb_map - rgb_gt)**2
        mse = resd_sq.mean()
        psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)

        if type == ImgLossType.PERC:
            if not hasattr(self, 'perc_loss_reference'):
                # For computing perceptual loss & img_loss # HACK: Nobody is referencing this as a pytorch module
                log('Initializing VGGPerceptualLoss')
                self.perc_loss_reference = [VGGPerceptualLoss().cuda().to(self.dtype)]  # move to specific location
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = self.perc_loss(rgb_map, rgb_gt)
        elif type == ImgLossType.CHARB: img_loss = (resd_sq + 0.001 ** 2).sqrt().mean()
        elif type == ImgLossType.HUBER: img_loss = huber(rgb_map, rgb_gt)
        elif type == ImgLossType.L2: img_loss = mse
        elif type == ImgLossType.L1: img_loss = l1(rgb_map, rgb_gt)
        elif type == ImgLossType.SSIM:
            rgb_gt = rgb_gt.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            rgb_map = rgb_map.view(-1, H, W, 3).permute(0, 3, 1, 2)  # B, C, H, W
            img_loss = 1. - compute_ssim(rgb_map, rgb_gt)
        else: raise NotImplementedError

        return psnr, img_loss

    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):

        # NOTE: a loss will be computed and logged if
        # 1. the corresponding loss weight is bigger than zero
        # 2. the corresponding components exist in the output
        def compute_image_loss(rgb_map: torch.Tensor, rgb_gt: torch.Tensor,
                               bg_color: torch.Tensor, msk_gt: torch.Tensor,
                               H: int = batch.meta.H[0].item(), W: int = batch.meta.W[0].item(),
                               type=self.img_loss_type):
            return self.compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W, type)

        if 'rgb_map' in output and \
           self.perc_loss_weight > 0 and \
           batch.meta.n_rays[0].item() == -1:
            if 'patch_h' in batch.meta and 'patch_w' in batch.meta:
                H, W = batch.meta.patch_h[0].item(), batch.meta.patch_w[0].item()
            else:
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            _, perc_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, H, W, type=ImgLossType.PERC)
            scalar_stats.perc_loss = perc_loss
            loss += self.perc_loss_weight * perc_loss

        if 'rgb_map' in output and \
           self.ssim_loss_weight > 0 and \
           batch.meta.n_rays[0].item() == -1:
            if 'patch_h' in batch.meta and 'patch_w' in batch.meta:
                H, W = batch.meta.patch_h[0].item(), batch.meta.patch_w[0].item()
            else:
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

            _, ssim_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk, H, W, type=ImgLossType.SSIM)
            scalar_stats.ssim_loss = ssim_loss
            loss += self.ssim_loss_weight * ssim_loss

        if 'rgb_map' in output and \
           self.img_loss_weight > 0:
            psnr, img_loss = compute_image_loss(output.rgb_map, batch.rgb, output.bg_color, batch.msk)
            scalar_stats.psnr = psnr
            scalar_stats.img_loss = img_loss
            loss += self.img_loss_weight * img_loss

        return loss

    # If we want to split the supervisor to include more sets of losses?
    def supervise(self, output: dotdict, batch: dotdict):
        loss = output.get('loss', 0)   # accumulated final loss
        loss_stats = output.get('loss_stats', dotdict())  # give modules ability to record something
        image_stats = output.get('image_stats', dotdict())  # give modules ability to record something
        scalar_stats = output.get('scalar_stats', dotdict())  # give modules ability to record something

        loss = self.compute_loss(output, batch, loss, scalar_stats, image_stats)

        for k, v in loss_stats.items():
            loss += v.mean()  # network computed loss

        output.loss = loss
        loss_stats.loss = loss  # these are the things to accumulated for loss
        scalar_stats.loss = loss  # these are the things to record and log
        return loss, scalar_stats, image_stats
