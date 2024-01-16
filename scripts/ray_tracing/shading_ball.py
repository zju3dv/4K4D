# this file is used for testing materials, thus should support large images files (8k maybe?)
# uniformly sample rays, no need for ray tracing since the ball should not be concerned with multi bounce shading
# currect techniques for static objects (articulated objects) including:
# 1. PRT (pre-computed radiance transfer), converts to a bunch of matrix dot product using spherical harmonics
# 2. Split-Sum for prefiltering the environment map and lighting, along with decomposed material node
# 3. Brute-Force shading with random sampling (this is what we're implementing) (used for ray tracing)

import math
import torch
import argparse
from tqdm import tqdm
from glob import glob

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log, color
from easyvolcap.utils.relight_utils import sample_envmap_image, read_hdr, Microfacet, linear2srgb, gen_light_xyz, gen_uniform_light_xyz
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.sh_utils import spher2cart, spherical_uniform_sampling_upper, spherical_uniform_sampling
from easyvolcap.utils.data_utils import save_image
# fmt: on


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--sample', type=int, default=10000, help='number of ray samples for the shaing results')
parser.add_argument('--fresnel', type=float, default=0.04)
parser.add_argument('--albedo', type=float, default=0.0)  # maybe a file for albedo map?
parser.add_argument('--roughness', type=float, default=0.3)  # maybe a file for roughness map?
parser.add_argument('--probe', type=str, default='data/lighting/8k/gym_entrance.hdr')
parser.add_argument('--output', type=str, default='data/shading_ball.png')

parser.add_argument('--stratified', action='store_true')
parser.add_argument('--env_h', type=int, default=16)
parser.add_argument('--env_w', type=int, default=32)
args = parser.parse_args()

# Prepare shapes
albedo, roughness, fresnel = args.albedo, args.roughness, args.fresnel
H, W, N = args.height, args.width, args.sample
log(f'Will produce a {color(f"{H}, {W}", "magenta")} shading ball with {color(f"{N}", "magenta")} samples for each pixel, albedo: {color(str(albedo), "magenta")}, roughness: {color(str(roughness), "magenta")}, fresnel: {color(str(fresnel), "magenta")}')

# Loading environment map for shading computation
log(f'Loading environment map from {color(args.probe, "blue")} onto {color(args.device, "magenta")}')
probe = torch.from_numpy(read_hdr(args.probe)).to(args.device, non_blocking=True)

# Construct the coordinate of the shading ball
zy = torch.stack(torch.meshgrid(torch.arange(H, device=args.device), torch.arange(W, device=args.device), indexing='ij'), dim=-1)
zy = zy - torch.tensor([H / 2, W / 2], device=zy.device)
zy = zy / torch.tensor(min(H, W) / 2, device=zy.device)
zy = zy.flip(dims=[0])
zy = zy.view(-1, 2)  # H * W, 2
x = (1 - zy.pow(2).sum(-1)).sqrt()

x = x.nan_to_num(0)
zyx = torch.cat([zy, x[..., None]], dim=-1)
surf = zyx.flip(dims=[-1])  # H * W, 3

# Construct normal and material for the shading ball
C = torch.tensor([2, 0, 0], device=x.device)  # simple perspecitve projection
norm = normalize(surf)  # whatever for invalid regions
view = normalize(surf - C)  # camera view direction

# Prepare mask for valid pixels
msk = (norm * view).sum(-1) < 0  # view direction and normal should be opposite
ind = msk.nonzero()
P = ind.shape[0]
surf = multi_gather(surf, ind)  # get good pixels to shade on
view = multi_gather(view, ind)  # get good pixels to shade on
norm = multi_gather(norm, ind)  # get good pixels to shade on


def image_based_lighting(surf: torch.Tensor,
                         norm: torch.Tensor,
                         view: torch.Tensor,
                         probe: dotdict,  # lighting
                         albedo: float,  # scalar albedo
                         roughness: float,  # scalar roughness
                         microfacet: Microfacet,  # material
                         N: int = 1024,  # number of samples
                         H: int = 16,
                         W: int = 32,
                         uniform: bool = True,  # uniform or stratified sampling
                         perturb: bool = True,
                         ):
    # Generate sample_count uniformly and stratified samples over the sphere
    P = surf.shape[0]
    if uniform:  # uniform random sampling
        T = P * N
        theta, phi = spherical_uniform_sampling_upper(T, device=surf.device)  # T, T,
        ray_d = spher2cart(theta, phi)  # T, 3, z always bigger than zero
    else:  # stratified sampling
        N = H * W
        T = P * N
        xyz, area = gen_light_xyz(H, W, device=surf.device)
        
        if perturb:
            R = torch.rand(3, 3, device=xyz.device)
            Q, R = torch.linalg.qr(R)  # 3, 3
            xyz = xyz @ Q  # apply random rotation
            
        xyz, area = xyz.view(-1, 3), area.view(-1, 1)
        ray_d = normalize(xyz)  # T, 3
        # Adding more samples seems to help, but not very good for low roughness surface (i.e. this implementation has a hard upper limit for specular surfaces)
        # And the visibility's influence is not clear enough, you've only done ablation on one of the char in one of the novel lighting
        # The physical correctness of distance field soft shadow is quesitonable

        # __import__('ipdb').set_trace()
        # torch.testing.assert_allclose(probe.view(-1, 3), sample_envmap_image(probe, ray_d))

        xyz = xyz[:, None].expand(N, P, 3).reshape(-1, 3)  # T, 3
        area = area[:, None].expand(N, P, 1).reshape(-1, 1)  # T, 1
        ray_d = ray_d[:, None].expand(N, P, 3).reshape(-1, 3)  # T, 3

    # Preparing shapes
    norm = norm[None].expand(N, P, 3).reshape(T, 3)  # T, 3
    view = view[None].expand(N, P, 3).reshape(T, 3)  # T, 3

    # Transform ray_d to be pointing upward from normal direction
    if uniform:
        R = torch.zeros([T, 3, 3], device=norm.device)
        R[..., 0, 0] = 1.0
        R[..., :3, 2] = norm  # c2w, z axis is normal direction
        R[..., :3, 1] = normalize(torch.cross(R[..., :3, 2], R[..., :3, 0]))
        R[..., :3, 0] = normalize(torch.cross(R[..., :3, 1], R[..., :3, 2]))
        ray_d = (R @ ray_d[..., None])[..., 0]

    # Compute shading
    ldot = (ray_d * norm).sum(dim=-1, keepdim=True)  # T
    light = sample_envmap_image(probe, ray_d)
    brdf = microfacet(ray_d, -view, norm, albedo, roughness)
    shading = light * ldot * brdf

    # Apply area to normalize integration
    if uniform:
        shading = shading * 2.0 * torch.pi / N
    else:
        shading = shading * area
    shading = shading.view(N, P, -1).sum(dim=-3)

    return shading


microfacet = Microfacet(f0=fresnel)

if not args.stratified:
    C = 2048
    rgb: torch.Tensor = 0
    for i in tqdm(range(0, N, C)):
        CC = min(N, i + C) - i  # the remaining chunk size (C or smaller)
        shading = image_based_lighting(surf, norm, view,
                                       probe, albedo, roughness, microfacet, CC)
        rgb = (rgb + shading)  # undo normalization and sum
    rgb = rgb * CC / N  # renormalization
else:
    N = math.ceil(N / (args.env_h * args.env_w))
    rgb: torch.Tensor = 0
    for i in tqdm(range(0, N, 1)):
        shading = image_based_lighting(surf, norm, view,
                                       probe, albedo, roughness, microfacet,
                                       H=args.env_h, W=args.env_w, uniform=False, perturb=N > 1)
        rgb = (rgb + shading)  # undo normalization and sum
    rgb = rgb * 1 / N  # renormalization

# Save rendered images
img = torch.zeros(H * W, 3, device=rgb.device)
img = multi_scatter(img, ind, rgb).view(H, W, 3)
img = linear2srgb(img)
img = torch.cat([img, msk.view(H, W, 1)], dim=-1)
save_image(args.output, img.detach().cpu().numpy())
