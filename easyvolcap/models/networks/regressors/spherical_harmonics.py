# Sepherical Harmonics in, view directions in, rgb out
# Need some way to pass extra view direction into this regressor, but how?
# Maybe just use the ray_d from batch..., as a somewhat ugly solution, MARK: This means no explicit chunking

# 1. Will try whether the input dir is None
# 2. Will try to use the last 3 dimensions of the input feature
# 3. If it's None, use batch.dir (stored by the user)
# 4. Or better yet, the user is responsible for this evaluation manually, sh_regressor only exists as a stub
import torch
from torch import nn
from easyvolcap.engine import REGRESSORS
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.models.networks.regressors.mlp_regressor import MlpRegressor


@REGRESSORS.register_module()
class SphericalHarmonics(nn.Module):
    def __init__(self,
                 sh_deg: int = 3,
                 in_dim: int = 256,
                 out_dim: int = 3,
                 out_actvn: nn.Module = nn.Sigmoid(),
                 mlp_regressor_cfg: dotdict = dotdict(type=MlpRegressor.__name__, backend='tcnn'),  # the humble sh parameter recovery system
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.sh_deg = sh_deg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sh_dim = (sh_deg + 1) ** 2 * out_dim
        self.out_actvn = out_actvn
        self.sh_mlp = REGRESSORS.build(mlp_regressor_cfg, in_dim=in_dim - 3, out_dim=self.sh_dim, out_actvn=nn.Identity(), **kwargs)  # HACK: hack to make network size match
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        keys = list(state_dict.keys())
        for key in keys:
            if f'{prefix}mlp_regressor' in key:
                state_dict[key.replace(f'{prefix}mlp_regressor', f'{prefix}sh_mlp')] = state_dict[key]
                del state_dict[key]

    def forward(self, feat: torch.Tensor, batch: dotdict = None, return_sh: bool = False):
        feat, dir = feat[..., :-3], feat[..., -3:]
        sh = self.sh_mlp(feat)
        sh = sh.view(*sh.shape[:-1], self.out_dim, self.sh_dim // self.out_dim)  # reshape to B, P, 3, SH
        if return_sh: return sh

        rgb = eval_sh(self.sh_deg, sh, dir)  # B, P, 3
        rgb = self.out_actvn(rgb)
        return rgb
