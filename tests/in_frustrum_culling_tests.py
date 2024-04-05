import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.math_utils import normalize, affine_inverse, affine_padding
from easyvolcap.utils.net_utils import typed, setup_deterministic
from easyvolcap.utils.data_utils import export_camera, export_pts
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera, render_fdgs, render_diff_gauss, in_frustrum
from easyvolcap.utils.console_utils import *


def test_gs_frustrum_culling():
    N = 100000

    K = torch.as_tensor([[736.5288696289062, 0.0, 682.7473754882812], [0.0, 736.4380493164062, 511.99737548828125], [0.0, 0.0, 1.0]], dtype=torch.float, device='cuda')
    R = torch.as_tensor([[0.9938720464706421, 0.0, -0.11053764075040817], [-0.0008741595083847642, 0.9999688267707825, -0.007859790697693825], [0.1105341762304306, 0.007908252067863941, 0.9938408732414246]], dtype=torch.float, device='cuda')
    T = torch.as_tensor([[-0.2975313067436218], [-1.2581647634506226], [0.2818146347999573]], dtype=torch.float, device='cuda')
    n = torch.as_tensor(2, dtype=torch.float, device='cuda')
    f = torch.as_tensor(3, dtype=torch.float, device='cuda')
    W = torch.as_tensor(1366, dtype=torch.long, device='cuda')
    H = torch.as_tensor(768, dtype=torch.long, device='cuda')
    camera = convert_to_gaussian_camera(K, R, T, H, W, n, f,
                                        K.cpu(), R.cpu(), T.cpu(), H.cpu(), W.cpu(), n.cpu(), f.cpu()
                                        )

    xyz = torch.rand(N, 3, device='cuda') * 10 - 5
    rgb = torch.rand(N, 3, device='cuda') * 10 - 5

    export_camera(affine_inverse(affine_padding(torch.cat([R, T], dim=-1))), K, filename='camera.ply')
    ray_o, ray_d = get_rays(H, W, K, R, T, z_depth=True, correct_pix=True)
    export_pts(ray_o + ray_d * n, filename='near_plane.ply')
    export_pts(ray_o + ray_d * f, filename='far_plane.ply')
    export_pts(xyz, rgb, filename='xyz.ply')
    ind = in_frustrum(xyz, camera.full_proj_transform).nonzero()[..., -1]
    export_pts(xyz[ind], rgb[ind], filename='in_frustrum.ply')


if __name__ == '__main__':
    my_tests(globals())
