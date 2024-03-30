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


def render_fdgs(xyz3: torch.Tensor, rgb3: torch.Tensor, cov: torch.Tensor, occ1: torch.Tensor, camera: dotdict):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
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
        debug=False
    )

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    scr = torch.zeros_like(xyz3, requires_grad=True) + 0  # gradient magic
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = typed(torch.float, torch.float)(rasterizer)(
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
    acc = torch.ones_like(rgb[..., :1])
    dpt = torch.zeros_like(rgb[..., :1])
    H = camera.image_height
    W = camera.image_width
    meta = dotdict({'radii': radii / float(max(H, W)), 'scr': scr, 'H': H, 'W': W})
    return rgb, acc, dpt, meta


def test_illegal_memory_diag():
    repeat = 100
    N = 10000
    cov = torch.rand(N, 6, device='cuda') - 0.5  # trying to reproduce this
    occ = torch.rand(N, 1, device='cuda')

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
        with open('fwd.dump', 'wb') as f:
            torch.save(dotdict(xyz3=xyz, rgb3=rgb, cov6=cov, occ1=occ, camera=camera), f)
            os.fsync(f)

        # render_fdgs(xyz, rgb, cov, occ, camera)  # will throw illegal memory access error
        render_diff_gauss(xyz, rgb, cov, occ, camera)  # will throw illegal memory access error


if __name__ == '__main__':
    setup_deterministic()
    my_tests(globals())
