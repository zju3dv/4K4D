import torch
from torch import nn
from torch.nn import functional as F
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor


@REGRESSORS.register_module()
class ImageBasedSphericalHarmonics(nn.Module):
    def __init__(self,
                 sh_deg: int = 3,
                 in_dim: int = 256 + 3,  # feature channel dim (vox + img?)
                 src_dim: int = 32 + 3,
                 out_dim: int = 3,
                 width: int = 64,
                 depth: int = 1,  # use small regressor network
                 resd_limit: float = 0.25,
                 resd_init: float = 0.0,
                 resd_weight_init: float = 0.01,
                 skip_shs: bool = False,
                 skip_eval_shs: bool = False,  # only skip the sh evaluation
                 blend_shs: bool = False,
                 manual_chunking: bool = False,

                 #  actvn: str = 'relu',
                 rgb_mlp_cfg: dotdict = dotdict(actvn='relu'),
                 sh_mlp_cfg: dotdict = dotdict(actvn='softplus'),
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.sh_deg = sh_deg
        self.sh_dim = (sh_deg + 1) ** 2 * out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skip_shs = skip_shs
        self.blend_shs = blend_shs
        self.skip_eval_shs = skip_eval_shs or skip_shs
        self.resd_limit = resd_limit
        self.manual_chunking = manual_chunking
        self.rgb_mlp = MlpRegressor(in_dim - 3 + src_dim, 1, width, depth, out_actvn=nn.Identity(), **rgb_mlp_cfg, **kwargs)  # blend weights
        if not self.skip_shs:
            if self.blend_shs:
                self.sh_mlp = MlpRegressor(in_dim - 3 + src_dim * 2, self.sh_dim, width, depth, out_actvn=nn.Identity(), **sh_mlp_cfg, **kwargs)
            else:
                self.sh_mlp = MlpRegressor(in_dim - 3, self.sh_dim, width, depth, out_actvn=nn.Identity(), **sh_mlp_cfg, **kwargs)
            if resd_init is not None:
                # TODO: Control the output magnitude
                [self.sh_mlp.mlp.linears[i].weight.data.normal_(0, resd_weight_init) for i in range(len(self.sh_mlp.mlp.linears))]
                [self.sh_mlp.mlp.linears[i].bias.data.fill_(resd_init if i == len(self.sh_mlp.mlp.linears) - 1 else -1) for i in range(len(self.sh_mlp.mlp.linears))]

    def forward(self, xyz_feat_dir: torch.Tensor, batch: dotdict, return_rgb_sh: bool = False):
        # geo_feat: B, P, C # vox(8) + img(16) + geo(64)?
        xyz_feat, dir = xyz_feat_dir[..., :-3], xyz_feat_dir[..., -3:]  # extract view direction

        # Prepare for directional feature
        ibr_feat: torch.Tensor = batch.output.src_feat_rgb  # B, S, P, C
        ibr_rgbs = ibr_feat[..., -3:]  # -4: dir feat, -7 -> -3: rgb, B, S, P, 3

        # Prepare for image based rendering blending (in a narrow sense)
        B, S, P, _ = ibr_feat.shape

        # Manual chunking, hopefully this will ease the memory usage
        if self.manual_chunking:
            rgb_bws = []
            for i in range(B):  # TODO: PERF
                for j in range(S):
                    feat = torch.cat([xyz_feat[i], ibr_feat[i, j]], dim=-1)  # P, C
                    rgb_bws.append(self.rgb_mlp(feat))  # P, 1
            rgb_bws = torch.stack(rgb_bws)  # BS, P, 1
            rgb_bws = rgb_bws.view(B, S, P, -1)  # B, S, P, 1
        else:
            exp_xyz_feat = xyz_feat[:, None].expand(ibr_feat.shape[:-1] + (xyz_feat.shape[-1], ))  # B, S, P, C
            feat = torch.cat([exp_xyz_feat, ibr_feat], dim=-1)  # +7, append the actual image feature
            rgb_bws = self.rgb_mlp(feat)

        rgb_bws = rgb_bws.softmax(-3)  # B, S, P, 1
        rgb = (ibr_rgbs * rgb_bws).sum(-3)  # B, P, 3, now we have the base rgb

        if not self.skip_shs:
            if self.blend_shs:
                sh = self.sh_mlp(torch.cat([xyz_feat, ibr_feat.var(1), ibr_feat.mean(1)], dim=-1))  # only available for geometry rendering
            else:
                sh = self.sh_mlp(xyz_feat)  # only available for geometry rendering
            sh = sh.view(*sh.shape[:-1], self.out_dim, self.sh_dim // self.out_dim)  # reshape to B, P, 3, SH
        else:
            sh = None
        if return_rgb_sh:
            return rgb, sh

        # Evaluation of specular SH and base image based rgb
        if not self.skip_eval_shs:
            rgb = rgb + eval_sh(self.sh_deg, sh, dir).tanh() * self.resd_limit  # B, P, 3
        rgb = rgb.clip(0, 1)  # maybe not needed?
        return rgb
