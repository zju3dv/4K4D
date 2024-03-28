import torch
from easyvolcap.utils.test_utils import my_tests
from easyvolcap.utils.net_utils import typed, setup_deterministic
from easyvolcap.utils.gaussian_utils import convert_to_gaussian_camera
from easyvolcap.utils.console_utils import *


def render_diff_gauss(xyz3: torch.Tensor, rgb3: torch.Tensor, cov: torch.Tensor, occ1: torch.Tensor, camera: dotdict):
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    # Prepare rasterization settings for gaussian
    raster_settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=camera.tanfovx,
        tanfovy=camera.tanfovy,
        bg=torch.full([3], 0.0, device=xyz3.device),  # GPU
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=0,
        campos=camera.camera_center,
        prefiltered=True,
        debug=False,
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = torch.zeros_like(xyz3, requires_grad=True) + 0  # gradient magic
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, rendered_depth, rendered_alpha, radii = typed(torch.float, torch.float)(rasterizer)(
        means3D=xyz3,
        means2D=scr,
        shs=None,
        colors_precomp=rgb3,
        opacities=occ1,
        scales=None,
        rotations=None,
        cov3D_precomp=cov,
    )

    rgb = rendered_image[None].permute(0, 2, 3, 1)
    acc = rendered_alpha[None].permute(0, 2, 3, 1)
    dpt = rendered_depth[None].permute(0, 2, 3, 1)
    H = camera.image_height
    W = camera.image_width
    meta = dotdict({'radii': radii / float(max(H, W)), 'scr': scr, 'H': H, 'W': W})
    return rgb, acc, dpt, meta


def test_illegal_memory_diag():
    repeat = 5
    N = 10000
    cov = torch.rand(N, 6, device='cuda') - 0.5  # trying to reproduce this
    occ = torch.rand(N, 1, device='cuda')

    # cov[:, 0] = 0
    # cov[:, 1] = -1
    # cov[:, 2] = 0
    # cov[:, 3] = 0
    # cov[:, 4] = 1
    # cov[:, 5] = 0

    K = torch.as_tensor([[736.5288696289062, 0.0, 682.7473754882812], [0.0, 736.4380493164062, 511.99737548828125], [0.0, 0.0, 1.0]], dtype=torch.float, device='cuda')
    R = torch.as_tensor([[0.9938720464706421, 0.0, -0.11053764075040817], [-0.0008741595083847642, 0.9999688267707825, -0.007859790697693825], [0.1105341762304306, 0.007908252067863941, 0.9938408732414246]], dtype=torch.float, device='cuda')
    T = torch.as_tensor([[-0.2975313067436218], [-1.2581647634506226], [0.2818146347999573]], dtype=torch.float, device='cuda')
    n = torch.as_tensor(0.02, dtype=torch.float, device='cuda')
    f = torch.as_tensor(1000.0, dtype=torch.float, device='cuda')
    W = torch.as_tensor(1366, dtype=torch.long, device='cuda')
    H = torch.as_tensor(768, dtype=torch.long, device='cuda')
    camera = convert_to_gaussian_camera(K, R, T, H, W, n, f,
                                        K.cpu(), R.cpu(), T.cpu(), H.cpu(), W.cpu(), n.cpu(), f.cpu()
                                        )

    for i in tqdm(range(repeat)):
        xyz = torch.rand(N, 3, device='cuda')
        rgb = torch.rand(N, 3, device='cuda')
        # with open('fwd.dump', 'wb') as f:
        #     torch.save(dotdict(xyz3=xyz, rgb3=rgb, cov6=cov, occ1=occ, camera=camera), f)
        #     os.fsync(f)
        if i == 4: breakpoint()
        render_diff_gauss(xyz, rgb, cov, occ, camera)  # will throw illegal memory access error


# def test_illegal_memory_rendering():
#     input = torch.load('fwd.dump')
#     input = dotdict(input)
#     # rgb, acc, dpt, meta = render_diff_gauss(input.xyz3, input.rgb3, input.cov6, input.occ1, input.camera)
#     N = len(input.xyz3)
#     # input.xyz3.requires_grad_()
#     # input.rgb3.requires_grad_()
#     # input.cov6.requires_grad_()
#     # input.occ1.requires_grad_()
#     first_N_points = 1
#     while first_N_points < 2 * N:
#         log(f'Rendering the first {min(N, first_N_points)} points out of {N}')
#         # for i in tqdm(range(repeat)):
#         rgb, acc, dpt, meta = render_diff_gauss(input.xyz3[:first_N_points], input.rgb3[:first_N_points], input.cov6[:first_N_points], input.occ1[:first_N_points], input.camera)
#         first_N_points *= 2


if __name__ == '__main__':
    setup_deterministic()
    my_tests(globals())
