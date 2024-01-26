import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.blend_utils import batch_rodrigues
from easyvolcap.utils.data_utils import to_x, add_batch, load_pts
from easyvolcap.utils.net_utils import make_buffer, make_params, typed
from easyvolcap.utils.math_utils import torch_inverse_2x2, point_padding


# def in_frustrum(xyz: torch.Tensor, ixt: torch.Tensor, ext: torch.Tensor):
def in_frustrum(xyz: torch.Tensor, full_proj_matrix: torch.Tensor, padding: float = 0.01):
    # __forceinline__ __device__ bool in_frustum(int idx,
    # 	const float* orig_points,
    # 	const float* viewmatrix,
    # 	const float* projmatrix,
    # 	bool prefiltered,
    # 	float3& p_view,
    # 	const float padding = 0.01f // padding in ndc space
    # 	)
    # {
    # 	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

    # 	// Bring points to screen space
    # 	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    # 	float p_w = 1.0f / (p_hom.w + 0.0000001f);
    # 	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
    # 	p_view = transformPoint4x3(p_orig, viewmatrix); // write this outside

    # 	// if (idx % 32768 == 0) printf("Viewspace point: %f, %f, %f\n", p_view.x, p_view.y, p_view.z);
    # 	// if (idx % 32768 == 0) printf("Projected point: %f, %f, %f\n", p_proj.x, p_proj.y, p_proj.z);
    # 	return (p_proj.z > -1 - padding) && (p_proj.z < 1 + padding) && (p_proj.x > -1 - padding) && (p_proj.x < 1. + padding) && (p_proj.y > -1 - padding) && (p_proj.y < 1. + padding);
    # }

    # xyz: N, 3
    # ndc = (xyz @ R.mT + T)[..., :3] @ K # N, 3
    # ndc[..., :2] = ndc[..., :2] / ndc[..., 2:] / torch.as_tensor([W, H], device=ndc.device) # N, 2, normalized x and y
    ndc = point_padding(xyz) @ full_proj_matrix
    ndc = ndc[..., :3] / ndc[..., 3:]
    return (ndc[..., 2] > -1 - padding) & (ndc[..., 2] < 1 + padding) & (ndc[..., 0] > -1 - padding) & (ndc[..., 0] < 1. + padding) & (ndc[..., 1] > -1 - padding) & (ndc[..., 1] < 1. + padding)  # N,


@torch.jit.script
def rgb2sh0(rgb: torch.Tensor):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


@torch.jit.script
def sh02rgb(sh: torch.Tensor):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@torch.jit.script
def get_jacobian(pix_xyz: torch.Tensor,  # B, P, 3, point in screen space
                 ):
    J = pix_xyz.new_zeros(pix_xyz.shape + (3, ))  # B, P, 3, 3
    J[..., 0, 0] = 1 / pix_xyz[..., 2]
    J[..., 1, 1] = 1 / pix_xyz[..., 2]
    J[..., 0, 2] = -pix_xyz[..., 0] / pix_xyz[..., 2]**2
    J[..., 1, 2] = -pix_xyz[..., 1] / pix_xyz[..., 2]**2
    J[..., 2, 2] = 1
    return J


@torch.jit.script
def gaussian_2d(xy: torch.Tensor,  # B, H, W, 2, screen pixel locations for evaluation
                mean_xy: torch.Tensor,  # B, H, W, K, 2, center of the gaussian in screen space
                cov_xy: torch.Tensor,  # B, H, W, 2, 2, covariance of the gaussian in screen space
                # pow: float = 1,  # when pow != 1, not a real gaussian, but easier to control fall off
                # we want to the values at 3 sigma to zeros -> easier to control volume rendering?
                ):
    inv_cov_xy = torch_inverse_2x2(cov_xy)  # B, P, 2, 2
    minus_mean = xy[..., None, :] - mean_xy  # B, P, K, 2
    # weight = torch.exp(-0.5 * torch.einsum('...d,...de,...e->...', x_minus_mean, inv_cov_xy, x_minus_mean))  # B, H, W, K
    xTsigma_new = (minus_mean[..., None] * inv_cov_xy[..., None, :, :]).sum(dim=-2)  # B, P, K, 2
    xTsigma_x = (xTsigma_new * minus_mean).sum(dim=-1)  # B, P, K
    return xTsigma_x


@torch.jit.script
def gaussian_3d(scale3: torch.Tensor,  # B, P, 3, the scale of the 3d gaussian in 3 dimensions
                rot3: torch.Tensor,  # B, P, 3, the rotation of the 3D gaussian (angle-axis)
                R: torch.Tensor,  # B, 3, 3, camera rotation
                ):
    sigma0 = torch.diag_embed(scale3)  # B, P, 3, 3
    rotmat = batch_rodrigues(rot3)  # B, P, 3, 3
    R_sigma = rotmat @ sigma0
    covariance = R @ R_sigma @ R_sigma.mT @ R.mT
    return covariance  # B, P, 3, 3


@torch.jit.script
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def strip_lowerdiag(L: torch.Tensor):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r: torch.Tensor):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.zeros((*sh, 4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    T[..., 3, 3] = 1.0
    return T


def getProjectionMatrix(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P


def prepare_gaussian_camera(batch):
    output = dotdict()
    H, W, K, R, T, n, f = batch.meta.H.item(), batch.meta.W.item(), batch.K[0], batch.R[0], batch.T[0], batch.meta.n.item(), batch.meta.f.item()

    output.image_height = H
    output.image_width = W

    output.K = K
    output.R = R
    output.T = T

    fl_x = batch.meta.K[0][0, 0]  # use cpu K
    fl_y = batch.meta.K[0][1, 1]  # use cpu K

    output.FoVx = focal2fov(fl_x, output.image_width)
    output.FoVy = focal2fov(fl_y, output.image_height)

    output.world_view_transform = getWorld2View(output.R, output.T).transpose(0, 1)
    output.projection_matrix = getProjectionMatrix(output.K, output.image_height, output.image_width, n, f).transpose(0, 1)
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix)
    output.camera_center = output.world_view_transform.float().inverse()[3:, :3].to(output.world_view_transform)

    # Set up rasterization configuration
    output.tanfovx = math.tan(output.FoVx * 0.5)
    output.tanfovy = math.tan(output.FoVy * 0.5)

    return output


def convert_to_gaussian_camera(K: torch.Tensor,
                               R: torch.Tensor,
                               T: torch.Tensor,
                               H: int,
                               W: int,
                               znear: float = 0.01,
                               zfar: float = 100.
                               ):
    output = dotdict()

    output.image_height = H
    output.image_width = W

    output.K = K
    output.R = R
    output.T = T

    fl_x = K[0, 0]
    fl_y = K[1, 1]

    output.FoVx = focal2fov(fl_x, output.image_width)
    output.FoVy = focal2fov(fl_y, output.image_height)

    output.world_view_transform = getWorld2View(output.R, output.T).transpose(0, 1)
    output.projection_matrix = getProjectionMatrix(output.K, output.image_height, output.image_width, znear, zfar).transpose(0, 1)
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix)  # 4, 4
    output.camera_center = output.world_view_transform.inverse()[3:, :3]

    # Set up rasterization configuration
    output.tanfovx = math.tan(output.FoVx * 0.5)
    output.tanfovy = math.tan(output.FoVy * 0.5)

    return output


class GaussianModel(nn.Module):
    def __init__(self,
                 xyz: torch.Tensor = None,
                 colors: torch.Tensor = None,
                 init_occ: float = 0.1,
                 init_scale: torch.Tensor = None,
                 sh_deg: int = 3,
                 scale_min: float = 1e-4,
                 scale_max: float = 1e1,
                 ):
        super().__init__()

        @torch.jit.script
        def scaling_activation(x, scale_min: float = scale_min, scale_max: float = scale_max):
            return torch.sigmoid(x) * (scale_max - scale_min) + scale_min

        @torch.jit.script
        def scaling_inverse_activation(x, scale_min: float = scale_min, scale_max: float = scale_max):
            return torch.logit(((x - scale_min) / (scale_max - scale_min)).clamp(1e-5, 1 - 1e-5))

        self.setup_functions(scaling_activation=scaling_activation, scaling_inverse_activation=scaling_inverse_activation)

        # SH realte configs
        self.active_sh_degree = make_buffer(torch.zeros(1))
        self.max_sh_degree = sh_deg

        # Initalize trainable parameters
        self.create_from_pcd(xyz, colors, init_occ, init_scale)

        # Densification related parameters
        self.max_radii2D = make_buffer(torch.zeros(self.get_xyz.shape[0]))
        self.xyz_gradient_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.denom = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))

        # Perform some model messaging before loading
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def setup_functions(self,
                        scaling_activation=torch.exp,
                        scaling_inverse_activation=torch.log,
                        opacity_activation=torch.sigmoid,
                        inverse_opacity_activation=inverse_sigmoid,
                        rotation_activation=F.normalize,
                        ):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = getattr(torch, scaling_activation) if isinstance(scaling_activation, str) else scaling_activation
        self.opacity_activation = getattr(torch, opacity_activation) if isinstance(opacity_activation, str) else opacity_activation
        self.rotation_activation = getattr(torch, rotation_activation) if isinstance(rotation_activation, str) else rotation_activation

        self.scaling_inverse_activation = getattr(torch, scaling_inverse_activation) if isinstance(scaling_inverse_activation, str) else scaling_inverse_activation
        self.opacity_inverse_activation = getattr(torch, inverse_opacity_activation) if isinstance(inverse_opacity_activation, str) else inverse_opacity_activation
        self.covariance_activation = build_covariance_from_scaling_rotation

    @property
    def device(self):
        return self.get_xyz.device

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, xyz: torch.Tensor, colors: torch.Tensor = None, opacities: float = 0.1, scales: torch.Tensor = None):
        from simple_knn._C import distCUDA2
        if xyz is None:
            xyz = torch.empty(0, 3, device='cuda')  # by default, init empty gaussian model on CUDA

        features = torch.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2))
        if colors is not None:
            SH = rgb2sh0(colors)
            features[:, :3, 0] = SH
        features[:, 3: 1:] = 0

        if scales is None:
            dist2 = torch.clamp_min(distCUDA2(xyz.float().cuda()), 0.0000001)
            scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        else:
            scales = self.scaling_inverse_activation(scales)

        rots = torch.rand((xyz.shape[0], 4))
        rots[:, 0] = 1

        if not isinstance(opacities, torch.Tensor) or len(opacities) != len(xyz):
            opacities = opacities * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        opacities = self.opacity_inverse_activation(opacities)

        self._xyz = make_params(xyz)
        self._features_dc = make_params(features[:, :, :1].transpose(1, 2).contiguous())
        self._features_rest = make_params(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scaling = make_params(scales)
        self._rotation = make_params(rots)
        self._opacity = make_params(opacities)

    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if prefix is not '' and not prefix.endswith('.'): prefix = prefix + '.'  # special care for when we're loading the model directly
        for name, params in self.named_parameters():
            if f'{prefix}{name}' in state_dict:
                params.data = params.data.new_empty(state_dict[f'{prefix}{name}'].shape)

    def reset_opacity(self, optimizer_state):
        for _, val in optimizer_state.items():
            if val.name == '_opacity':
                break
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        self._opacity.set_(opacities_new.detach())
        self._opacity.grad = None
        val.old_keep = torch.zeros_like(val.old_keep, dtype=torch.bool)
        val.new_keep = torch.zeros_like(val.new_keep, dtype=torch.bool)
        val.new_params = self._opacity
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        # optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self._xyz.set_(self._xyz[valid_points_mask].detach())
        self._xyz.grad = None
        self._features_dc.set_(self._features_dc[valid_points_mask].detach())
        self._features_dc.grad = None
        self._features_rest.set_(self._features_rest[valid_points_mask].detach())
        self._features_rest.grad = None
        self._opacity.set_(self._opacity[valid_points_mask].detach())
        self._opacity.grad = None
        self._scaling.set_(self._scaling[valid_points_mask].detach())
        self._scaling.grad = None
        self._rotation.set_(self._rotation[valid_points_mask].detach())
        self._rotation.grad = None

        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[valid_points_mask])
        self.xyz_gradient_accum.grad = None
        self.denom.set_(self.denom[valid_points_mask])
        self.denom.grad = None
        self.max_radii2D.set_(self.max_radii2D[valid_points_mask])
        self.max_radii2D.grad = None

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer_state):
        d = dotdict({
            "_xyz": new_xyz,
            "_features_dc": new_features_dc,
            "_features_rest": new_features_rest,
            "_opacity": new_opacities,
            "_scaling": new_scaling,
            "_rotation": new_rotation,
        })

        # optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        for name, new_params in d.items():
            params: nn.Parameter = getattr(self, name)
            params.set_(torch.cat((params.data, new_params), dim=0).detach())
            params.grad = None

        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.xyz_gradient_accum.grad = None
        self.denom.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.grad = None
        self.max_radii2D.set_(torch.zeros((self.get_xyz.shape[0]), device=device))
        self.max_radii2D.grad = None

        for val in optimizer_state.values():
            name = val.name
            val.new_keep = torch.cat((val.new_keep, torch.zeros_like(d[name], dtype=torch.bool, requires_grad=False)), dim=0)
            val.new_params = getattr(self, name)
            assert val.new_keep.shape == val.new_params.shape

    def densify_and_split(self, grads, grad_threshold, scene_extent, percent_dense, min_opacity, max_screen_size, optimizer_state, N=2):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, optimizer_state)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(prune_filter)
        old_keep_mask = ~prune_filter[:grads.shape[0]]
        for val in optimizer_state.values():
            name = val.name
            val.old_keep[~old_keep_mask] = False
            val.new_keep = val.new_keep[~prune_filter]
            val.params = getattr(self, name)
            assert val.old_keep.sum() == val.new_keep.sum()
            assert val.new_keep.shape == val.new_params.shape

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        _old_keep_mask = old_keep_mask.clone()
        mask_mask = old_keep_mask[old_keep_mask]
        _mask = prune_mask[:mask_mask.shape[0]]
        mask_mask[_mask] = False
        old_keep_mask[_old_keep_mask] = mask_mask
        for val in optimizer_state.values():
            name = val.name
            val.old_keep[~old_keep_mask] = False
            val.new_keep = val.new_keep[~prune_mask]
            val.params = getattr(self, name)
            assert val.old_keep.sum() == val.new_keep.sum()
            assert val.new_keep.shape == val.new_params.shape

    def densify_and_clone(self, grads, grad_threshold, scene_extent, percent_dense, optimizer_state):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, optimizer_state)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, percent_dense, optimizer_state):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, percent_dense, optimizer_state)
        self.densify_and_split(grads, max_grad, extent, percent_dense, min_opacity, max_screen_size, optimizer_state)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        from plyfile import PlyData, PlyElement
        os.makedirs(dirname(path), exist_ok=True)

        # The original gaussian model uses a different activation
        # Normalization for rotation, so no conversion
        # Exp on scaling, need to -> world space -> log

        # Doing inverse_sigmoid here will lead to NaNs
        opacity = self._opacity
        if self.opacity_activation != F.sigmoid and \
                self.opacity_activation != torch.sigmoid and \
                not isinstance(self.opacity_activation, nn.Sigmoid):
            opacity = self.opacity_activation(opacity)
            _opacity = inverse_sigmoid(opacity)

        scale = self._scale
        scale = self.scaling_activation(scale)
        _scale = torch.log(scale)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = _opacity.detach().cpu().numpy()
        scale = _scale.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str):
        xyz, _, _, scalars = load_pts(path)

        # The original gaussian model uses a different activation
        xyz = torch.from_numpy(xyz)
        rotation = torch.from_numpy(np.concatenate([scalars['rot_{}'.format(i)] for i in range(4)], axis=-1))
        scaling = torch.from_numpy(np.concatenate([scalars['scale_{}'.format(i)] for i in range(3)], axis=-1))
        scaling = torch.exp(scaling)
        scaling = self.scaling_inverse_activation(scaling)
        opacity = torch.from_numpy(scalars['opacity'])

        # Doing inverse_sigmoid here will lead to NaNs
        if self.opacity_activation != F.sigmoid and \
                self.opacity_activation != torch.sigmoid and \
                not isinstance(self.opacity_activation, nn.Sigmoid):
            opacity = inverse_sigmoid(opacity)
            opacity = self.opacity_inverse_activation(opacity)

        # Load the SH colors
        features_dc = torch.empty((xyz.shape[0], 3, 1))
        features_dc[:, 0] = torch.from_numpy(np.asarray(scalars["f_dc_0"]))
        features_dc[:, 1] = torch.from_numpy(np.asarray(scalars["f_dc_1"]))
        features_dc[:, 2] = torch.from_numpy(np.asarray(scalars["f_dc_2"]))

        extra_f_names = [k for k in scalars.keys() if k.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_rest = torch.zeros((xyz.shape[0], len(extra_f_names), 1))
        for idx, attr_name in enumerate(extra_f_names):
            features_rest[:, idx] = torch.from_numpy(np.asarray(scalars[attr_name]))
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_rest = features_rest.view(features_rest.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)

        state_dict = dotdict()
        state_dict._xyz = xyz
        state_dict._features_dc = features_dc.mT
        state_dict._features_rest = features_rest.mT
        state_dict._opacity = opacity
        state_dict._scaling = scaling
        state_dict._rotation = rotation

        self.load_state_dict(state_dict, strict=False)
        self.active_sh_degree.data.fill_(self.max_sh_degree)

    def render(self, batch: dotdict):
        # TODO: Make rendering function easier to read, now there're at least 3 types of gaussian rendering function
        from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

        # Prepare renderable parameters, without batch
        xyz = self.get_xyz
        scale3 = self.get_scaling
        rot4 = self.get_rotation
        occ = self.get_opacity
        sh = self.get_features

        # Prepare the camera transformation for Gaussian
        gaussian_camera = to_x(prepare_gaussian_camera(batch), torch.float)

        # is_in_frustrum = in_frustrum(xyz, gaussian_camera.full_proj_transform)
        # print('Number of points to render:', is_in_frustrum.sum().item())

        # Prepare rasterization settings for gaussian
        raster_settings = GaussianRasterizationSettings(
            image_height=gaussian_camera.image_height,
            image_width=gaussian_camera.image_width,
            tanfovx=gaussian_camera.tanfovx,
            tanfovy=gaussian_camera.tanfovy,
            bg=torch.full([3], 0.0, device=xyz.device),  # GPU # TODO: make these configurable
            scale_modifier=1.0,  # TODO: make these configurable
            viewmatrix=gaussian_camera.world_view_transform,
            projmatrix=gaussian_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=gaussian_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        scr = torch.zeros_like(xyz, requires_grad=True) + 0  # gradient magic
        if scr.requires_grad: scr.retain_grad()
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered_image, rendered_depth, rendered_alpha, radii = typed(torch.float, torch.float)(rasterizer)(
            means3D=xyz,
            means2D=scr,
            shs=sh,
            colors_precomp=None,
            opacities=occ,
            scales=scale3,
            rotations=rot4,
            cov3D_precomp=None,
        )

        # No batch dimension
        rgb = rendered_image.permute(1, 2, 0)
        acc = rendered_alpha.permute(1, 2, 0)
        dpt = rendered_depth.permute(1, 2, 0)

        return rgb, acc, dpt  # H, W, C


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_depth, rendered_alpha, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return dotdict({
        "render": rendered_image,
        "alpha": rendered_alpha,
        "depth": rendered_depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    })


def naive_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, radii, rendered_depth = rasterizer(
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer.raster_settings = raster_settings

    colors_precomp = torch.ones_like(means3D, requires_grad=False).contiguous()
    rendered_alpha, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    colors_precomp = F.pad(means3D, (0, 1), value=1.0) @ viewpoint_camera.world_view_transform
    colors_precomp = torch.norm(colors_precomp[..., :3] - viewpoint_camera.camera_center, dim=-1, keepdim=True)
    colors_precomp = torch.repeat_interleave(colors_precomp, 3, dim=-1).contiguous()
    rendered_depth, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return dotdict({
        "render": rendered_image[:3],
        "alpha": rendered_alpha[:1],
        "depth": rendered_depth[:1],
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii
    })
