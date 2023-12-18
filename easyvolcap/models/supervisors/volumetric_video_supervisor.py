# Default loss module (called supervisor)
import torch
import numpy as np
from torch import nn
from typing import Union
import math

from easyvolcap.engine import SUPERVISORS
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.net_utils import multi_gather, multi_scatter, resize_image, VolumetricVideoModule, ray2xyz, freeze_module, unfreeze_module, schlick_bias
from easyvolcap.utils.loss_utils import mse, eikonal, l1, l2, l2_reg, l1_reg, lossfun_distortion, lossfun_outer, huber, mIoU_loss, bce_loss, elastic_crit, VGGPerceptualLoss, PerceptualLoss, ImgLossType, ElasticLossReduceType, general_loss_with_squared_residual, compute_planes_tv, compute_time_planes_smooth, lossfun_zip_outer, compute_ssim
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork


@SUPERVISORS.register_module()
class VolumetricVideoSupervisor(VolumetricVideoModule):
    def __init__(self,
                 network: nn.Module,
                 img_loss_weight: float = 1.0,
                 msk_loss_weight: float = 0.0,
                 prop_loss_weight: float = 1.0,
                 zip_prop_loss_weight: float = 0.0,
                 dist_loss_weight: float = 0.0,
                 elas_loss_weight: float = 0.0,
                 perc_loss_weight: float = 0.0,  # smaller loss on perc
                 ssim_loss_weight: float = 0.0,
                 resd_loss_weight: float = 0.0,
                 msk_mse_weight: float = 0.0,  # mask mse weight
                 vq_loss_weight: float = 0.0,

                 tv_loss_weight: float = 0.0,  # total variation loss for kplanes # FIXME: only for kplanes
                 time_smooth_weight: float = 0.0,  # time smooth weight # FIXME: only for kplanes
                 time_smooth_prop_weight: float = 0.0,  # time smooth propasal weight # FIXME: only for kplanes
                 pvdiff_loss_weight: float = 0.0,  # point cloud and volume density loss for volumetric video
                 pcd_sparse_loss_weight: float = 0.0,  # point cloud sparsity (distances)
                 pcd_sparse_K: int = 5,

                 conf_sparse_loss_weight: float = 0.0,

                 sparse_loss_weight: float = 0.0,  # sparsity loss (plenoctree)
                 n_sparse_points: int = 65536,  # number of points to sample for sparsity loss
                 sparse_dist: float = 0.005,

                 img_loss_type: ImgLossType = ImgLossType.HUBER.name,  # chabonier loss for img_loss
                 elas_reduce_type: ElasticLossReduceType = ElasticLossReduceType.WEIGHT.name,

                 anneal_slope: float = 10.0,  # annealing function
                 base_resd_weight: float = 1.0,  # residual should be small
                 dtype: Union[str, torch.dtype] = torch.float,

                 # SDF model
                 eikonal_loss_weight: float = 0.0,
                 curvature_loss_weight: float = 0.0,

                 #  # iavatar model
                 #  hard_surface_loss_weight: float = 0.0,

                 #  # meshsdf model
                 #  smpl_guided_loss_weight: float = 0.0,
                 #  mesh_surface_loss_weight: float = 0.0,
                 ):
        super().__init__(network)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.anneal_slope = anneal_slope

        self.pvdiff_loss_weight = pvdiff_loss_weight
        self.pcd_sparse_loss_weight = pcd_sparse_loss_weight
        self.pcd_sparse_K = pcd_sparse_K

        self.img_loss_weight = img_loss_weight
        self.img_loss_type = ImgLossType[img_loss_type]

        # For computing perceptual loss & img_loss # HACK: Nobody is referencing this as a pytorch module
        self.perc_loss_weight = perc_loss_weight
        if self.perc_loss_weight > 0: self.perc_loss_reference = [VGGPerceptualLoss().cuda().to(self.dtype)]  # move to specific location

        self.ssim_loss_weight = ssim_loss_weight

        self.msk_loss_weight = msk_loss_weight
        self.msk_mse_weight = msk_mse_weight
        self.vq_loss_weight = vq_loss_weight
        self.prop_loss_weight = prop_loss_weight
        self.zip_prop_loss_weight = zip_prop_loss_weight
        self.dist_loss_weight = dist_loss_weight
        self.resd_loss_weight = resd_loss_weight
        self.base_resd_weight = base_resd_weight
        self.tv_loss_weight = tv_loss_weight
        self.time_smooth_weight = time_smooth_weight
        self.time_smooth_prop_weight = time_smooth_prop_weight

        self.elas_loss_weight = elas_loss_weight
        self.elas_reduce_type = ElasticLossReduceType[elas_reduce_type]

        self.sparse_loss_weight = sparse_loss_weight
        self.n_sparse_points = n_sparse_points
        self.sparse_dist = sparse_dist
        self.conf_sparse_loss_weight = conf_sparse_loss_weight

        self.eikonal_loss_weight = eikonal_loss_weight
        self.curvature_loss_weight = curvature_loss_weight

        # self.hard_surface_loss_weight = hard_surface_loss_weight

        # self.smpl_guided_loss_weight = smpl_guided_loss_weight
        # self.mesh_surface_loss_weight = mesh_surface_loss_weight

        self.forward = self.supervise

    @property
    def perc_loss(self):
        return self.perc_loss_reference[0]

    def supervise(self, output: dotdict, batch: dotdict):

        loss = 0  # accumulated final loss
        image_stats = output.get('image_stats', dotdict())  # give modules ability to record something
        scalar_stats = output.get('scalar_stats', dotdict())  # give modules ability to record something

        def compute_image_loss(rgb_map: torch.Tensor, rgb_gt: torch.Tensor,
                               bg_color: torch.Tensor, msk_gt: torch.Tensor,
                               H=batch.meta.H[0].item(), W=batch.meta.W[0].item(),
                               type=self.img_loss_type):
            rgb_gt = rgb_gt * msk_gt + bg_color * (1 - msk_gt)  # MARK: modifying gt for supervision

            # https://stackoverflow.com/questions/181530/styling-multi-line-conditions-in-if-statements
            resd_sq = (rgb_map - rgb_gt)**2
            mse = resd_sq.mean()
            psnr = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)

            if type == ImgLossType.PERC:
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

        # NOTE: a loss will be computed and logged if
        # 1. the corresponding loss weight is bigger than zero
        # 2. the corresponding components exist in the output
        if 's_vals' in output and 'weights' in output and \
           self.dist_loss_weight > 0:
            dist_loss = 0.0
            dist_loss += lossfun_distortion(output.s_vals, output.weights).mean()

            if 's_vals_prop' in output and 'weights_prop' in output:
                for i in range(len(output.s_vals_prop)):
                    dist_loss += lossfun_distortion(output.s_vals_prop[i], output.weights_prop[i]).mean()

            scalar_stats.dist_loss = dist_loss
            loss += self.dist_loss_weight * dist_loss

        if 'rgb_maps_prop' in output and 'bg_colors_prop' in output and \
           len(output.rgb_maps_prop) and len(output.bg_colors_prop) and \
           self.prop_loss_weight > 0:
            prop_loss = 0
            for i in range(len(output.rgb_maps_prop)):
                rgb_gt = batch.rgb
                msk_gt = batch.msk
                rgb_map = output.rgb_maps_prop[i]
                bg_color = output.bg_colors_prop[i]
                H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
                if 'Hrs_prop' in output and 'Wrs_prop' in output:
                    B = rgb_map.shape[0]
                    Hr, Wr = output.Hrs_prop[i], output.Wrs_prop[i]
                    rgb_gt = resize_image(rgb_gt.view(B, H, W, -1), size=(Hr, Wr)).view(B, Hr * Wr, -1)  # B, P, 3
                    msk_gt = resize_image(msk_gt.view(B, H, W, -1), size=(Hr, Wr)).view(B, Hr * Wr, -1)  # B, P, 3
                    H, W = Hr, Wr
                _, prop_loss_i = compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W)
                prop_loss += prop_loss_i

                if self.perc_loss_weight != 0:
                    _, prop_perc_loss_i = compute_image_loss(rgb_map, rgb_gt, bg_color, msk_gt, H, W, type=ImgLossType.PERC)
                    prop_loss += prop_perc_loss_i * self.perc_loss_weight

            prop_loss = prop_loss / len(output.rgb_maps_prop)
            scalar_stats.prop_loss = prop_loss
            loss += self.prop_loss_weight * prop_loss

        if 'jacobian_prop' in output and 'weights_prop' in output and \
           len(output.jacobian_prop) and len(output.weights_prop) and \
           self.elas_loss_weight > 0:
            prop_elas_loss = 0
            for i in range(len(output.jacobian_prop)):
                jacobian = output.jacobian_prop[i]
                weights = output.weights_prop[i]
                prop_elas_loss_i = elastic_crit(jacobian)  # (B, N)
                if self.elas_reduce_type == ElasticLossReduceType.WEIGHT:
                    prop_elas_loss_i = prop_elas_loss_i * weights.reshape(output.weights.shape[0], -1)
                prop_elas_loss_i = torch.sum(prop_elas_loss_i, dim=-1).mean()
                prop_elas_loss += prop_elas_loss_i
            scalar_stats.prop_elas_loss = prop_elas_loss
            loss += self.elas_loss_weight * prop_elas_loss

        if 'resd_prop' in output and len(output.resd_prop) and \
           self.resd_loss_weight > 0:
            prop_resd_loss = 0
            for i in range(len(output.resd_prop)):
                prop_resd_loss_i = (output.resd_prop[i] ** 2).mean()
                prop_resd_loss += prop_resd_loss_i
            scalar_stats.prop_resd_loss = prop_resd_loss
            loss += self.resd_loss_weight * prop_resd_loss

        if 's_vals_prop' in output and 'weights_prop' in output and \
           len(output.s_vals_prop) and len(output.weights_prop) and \
           's_vals' in output and 'weights' in output and \
           self.prop_loss_weight > 0:
            prop_loss = 0
            for i in range(len(output.s_vals_prop)):
                prop_loss += lossfun_outer(
                    output.s_vals.detach(), output.weights.detach(),
                    output.s_vals_prop[i], output.weights_prop[i])
            prop_loss = prop_loss.mean()
            scalar_stats.prop_loss = prop_loss
            loss += self.prop_loss_weight * prop_loss

        if 's_vals_prop' in output and 'weights_prop' in output and \
           len(output.s_vals_prop) and len(output.weights_prop) and \
           's_vals' in output and 'weights' in output and \
           self.zip_prop_loss_weight > 0:
            zip_prop_loss = 0
            pulse_width = [0.03, 0.003, 0.0003]
            for i in range(len(output.s_vals_prop)):
                zip_prop_loss += lossfun_zip_outer(
                    output.s_vals.detach(), output.weights.detach(),
                    output.s_vals_prop[i], output.weights_prop[i],
                    pulse_width=pulse_width[i])
            scalar_stats.zip_prop_loss = zip_prop_loss
            loss += self.zip_prop_loss_weight * zip_prop_loss

        if 'acc_map' in output and 'msk' in batch and \
           self.msk_loss_weight > 0:
            msk_loss = mIoU_loss(output.acc_map, batch.msk)
            scalar_stats.msk_loss = msk_loss
            loss += self.msk_loss_weight * msk_loss

        if 'acc_map' in output and 'msk' in batch and \
           self.msk_mse_weight > 0:
            msk_loss = mse(output.acc_map, batch.msk)
            scalar_stats.msk_mse = msk_loss
            loss += self.msk_mse_weight * msk_loss

        if self.tv_loss_weight > 0:
            tv_loss = 0
            for net in self.network.networks:
                tv_loss += compute_planes_tv(net.xyzt_embedder.spatial_embedding)
                # if net.xyzt_embedder.temporal_embedding[0].shape[-1]>3: # only add tv loss for temporal embedding if there are more than 3 planes
                # tv_loss += compute_planes_tv(net.xyzt_embedder.temporal_embedding)
            scalar_stats.tv_loss = tv_loss
            loss += self.tv_loss_weight * tv_loss

        if self.time_smooth_weight > 0:
            net = self.network.networks[-1]
            time_smooth_loss = compute_time_planes_smooth(net.xyzt_embedder.temporal_embedding)
            scalar_stats.time_smooth_loss = time_smooth_loss
            loss += self.time_smooth_weight * time_smooth_loss

        if self.time_smooth_prop_weight > 0:
            time_smooth_loss = 0
            for net in self.network.networks[:-1]:
                time_smooth_loss = compute_time_planes_smooth(net.xyzt_embedder.temporal_embedding)
            scalar_stats.time_smooth_prop_loss = time_smooth_loss
            loss += self.time_smooth_prop_weight * time_smooth_loss

        if 'jacobian' in output and 'weights' in output and 'jacobian_prop' not in output and \
           self.elas_loss_weight > 0:
            elas_loss = elastic_crit(output.jacobian)   # (B, N)
            if self.elas_reduce_type == ElasticLossReduceType.WEIGHT:
                elas_loss = elas_loss * output.weights.reshape(output.weights.shape[0], -1)
            elas_loss = torch.sum(elas_loss, dim=-1).mean()
            scalar_stats.elas_loss = elas_loss
            loss += self.elas_loss_weight * elas_loss

        # residual displacement supervision for dnerf
        if 'resd' in output and 'resd_prop' not in output and \
           self.resd_loss_weight > 0:
            resd_loss = l2_reg(output.resd)
            scalar_stats.resd_loss = resd_loss
            loss += self.resd_loss_weight * resd_loss

        # residual displacement supervision for dnerf
        if 'base_resd' in output and \
           self.base_resd_weight > 0:
            base_resd = l2_reg(output.base_resd)
            scalar_stats.base_resd = base_resd
            loss += self.base_resd_weight * base_resd

        # residual displacement supervision for dnerf
        if 'spec_resd' in output and \
           self.base_resd_weight > 0:
            spec_resd = l2_reg(output.spec_resd)
            scalar_stats.spec_resd = spec_resd
            loss += self.base_resd_weight * spec_resd

        if self.sparse_loss_weight > 0:
            t: torch.Tensor = batch.t
            bounds: torch.Tensor = batch.bounds  # B, 2, 3
            batch.output = output  # remember these for forward
            xyz = torch.rand(bounds.shape[0], self.n_sparse_points, 3, dtype=bounds.dtype, device=bounds.device)
            xyz = xyz * (bounds[..., 1, :] - bounds[..., 0, :]) + bounds[..., 0, :]
            occ = self.network.occ(xyz, t, self.sparse_dist, batch)  # B, N, 1
            sparse_loss = occ.mean()
            scalar_stats.sparse_loss = sparse_loss
            loss += self.sparse_loss_weight * sparse_loss

        if self.conf_sparse_loss_weight > 0 and \
           'confidence' in output:
            confidence: torch.Tensor = output.confidence.view(-1)  # B, S, K, 1 -> N,
            conf_sparse_loss = (confidence.log() + (1 - confidence).log()).sum() / confidence.norm()

            scalar_stats.conf_sparse_loss = conf_sparse_loss
            loss += self.conf_sparse_loss_weight * conf_sparse_loss

        if self.pvdiff_loss_weight > 0 and 'pcd_w_grad' in output:
            ratio = schlick_bias(batch.meta.frac, self.anneal_slope)

            # How do we optimize point locations manually without optimizing the network?
            pcd = output.pcd_w_grad
            freeze_module(self.network)
            occ = self.network.occ(pcd, batch.t, 0.005, batch)  # B, N, 1 # the dist parameter is a placeholder
            unfreeze_module(self.network)  # will this do?
            pvdiff_loss = 1 - occ.mean()  # make the gradient backward
            scalar_stats.pvdiff_loss = pvdiff_loss
            loss += ratio * self.pvdiff_loss_weight * pvdiff_loss

        if self.pcd_sparse_loss_weight > 0 and 'pcd_w_grad' in output:
            ratio = schlick_bias(batch.meta.frac, self.anneal_slope)

            # Move points away from each other
            pcd = output.pcd_w_grad
            K = self.pcd_sparse_K
            B, S, C = pcd.shape
            B, N, C = pcd.shape
            from pytorch3d.ops import knn_points

            close = knn_points(pcd, pcd, K=K)
            dists, idx = close.dists, close.idx
            pcd_new = multi_gather(pcd, idx.view(B, S * K)).view(B, S, K, -1)  # B, S, K, 3
            pcd_sparse_loss = mse(pcd, pcd_new.mean(dim=-2))
            scalar_stats.pcd_sparse_loss = pcd_sparse_loss
            loss += ratio * self.pcd_sparse_loss_weight * pcd_sparse_loss

        if self.vq_loss_weight > 0.0 and 'vq_loss' in output:
            vq_loss = output.vq_loss.mean()  # average over batch
            scalar_stats.vq_loss = vq_loss
            loss += self.vq_loss_weight * vq_loss  # already premultiplied by the loss weight

        # log(batch.meta.n_rays)

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

        if self.eikonal_loss_weight > 0.0 and 'gradients' in output:
            gradients = output.gradients
            eikonal_loss = eikonal(gradients)
            scalar_stats.eikonal_loss = eikonal_loss
            loss += self.eikonal_loss_weight * eikonal_loss

        if self.curvature_loss_weight > 0.0 and 'sampled_sdf' in output:
            delta = self.network.finite_diff_delta
            centered_sdf = output.sdf
            sourounding_sdf = output.sampled_sdf
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta ** 2)
            curvature_loss = curvature.abs().mean() * self.network.curvature_loss_multi_factor
            scalar_stats.curvature_loss = curvature_loss
            loss += self.curvature_loss_weight * curvature_loss

        # if self.hard_surface_loss_weight > 0.0 and 'weights' in output:
        #     const = 0.313262
        #     hard_surface_weights = (-torch.log(torch.exp(-output.weights) + torch.exp(output.weights - 1))).mean() + const
        #     hard_surface_alpha = (-torch.log(torch.exp(-output.acc_map) + torch.exp(output.acc_map - 1))).mean() + const
        #     hard_surface_loss = hard_surface_weights + hard_surface_alpha
        #     scalar_stats.hard_surface_loss = hard_surface_loss
        #     loss += self.hard_surface_loss_weight * hard_surface_loss

        # if self.smpl_guided_loss_weight > 0.0 and 'smpl_xyz' in output and 'mesh_xyz' in output.keys():
        #     smpl_guided_loss = (output.smpl_xyz - output.mesh_xyz).square().sum(dim=-1).mean()
        #     scalar_stats.smpl_guided_loss = smpl_guided_loss
        #     loss += self.smpl_guided_loss_weight * smpl_guided_loss

        # if self.mesh_surface_loss_weight > 0.0 and 'mesh_sdf' in output:
        #     mesh_surface_loss = output.mesh_sdf.square().mean()
        #     scalar_stats.mesh_surface_loss = mesh_surface_loss
        #     loss += self.mesh_surface_loss_weight * mesh_surface_loss

        # NeuS related scalars
        if 'deviation' in output:
            scalar_stats.deviation = output.deviation.mean()

        # VolSDF related scalars
        if 'beta' in output:
            scalar_stats.beta = output.beta.mean()

        scalar_stats.loss = loss  # also record the total loss
        return loss, scalar_stats, image_stats
