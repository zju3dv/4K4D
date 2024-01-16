# this file loads obj with textures and mlt files, then perform rasteization with ray traced shadow
# it can also render components like normal, albedo, roughness, visibility etc.
# this should only be used for rendering ground truth values to compute metrics
# for visibility, we should compute metrics on visibility or the whole shading? only for soft-shadow?
# maybe both would be better...
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join
from termcolor import colored

from bvh_ray_tracing import BVH
from pytorch3d.structures import Meshes

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.chunk_utils import multi_gather_tris
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_mesh, save_unchanged
from easyvolcap.utils.relight_utils import read_hdr, sample_envmap_image
from easyvolcap.utils.sh_utils import spher2cart, spherical_uniform_sampling_upper
from easyvolcap.utils.raster_utils import render_nvdiffrast, get_ndc_perspective_matrix


def light_visibility(surf: torch.Tensor,  # B, P, 3
                     norm: torch.Tensor,  # B, P, 3
                     tris: torch.Tensor,
                     N: int = 100,  # number of samples per pixel (randomly distribute on sphere)
                     compute_lvis: bool = True,
                     ):  # this function will compute both lvis and ldot
    # Prepare shapes of verts and faces (could have same batch size as surf and norm)
    if surf.ndim == 4:
        surf = surf.view(surf.shape[0], -1, 3)
    if norm.ndim == 4:
        norm = norm.view(surf.shape[0], -1, 3)
    B, P, _ = surf.shape
    T = B * N * P

    # Generate sample_count uniformly and stratified samples over the sphere
    # See http://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
    theta, phi = spherical_uniform_sampling_upper(T, device=surf.device)  # T, T,
    ray_d = spher2cart(theta, phi)  # T, 3, z always bigger than zero

    # Preparing shapes
    norm = norm[:, None].expand(B, N, P, 3).reshape(T, 3)  # T, 3
    ray_o = surf[:, None].expand(B, N, P, 3).reshape(T, 3)  # T, 3

    # Transform ray_d to be pointing upward from normal direction
    R = torch.zeros([T, 3, 3], device=norm.device)
    R[..., 0, 0] = 1.0
    R[..., :3, 2] = norm  # c2w, z axis is normal direction
    R[..., :3, 1] = normalize(torch.cross(R[..., :3, 2], R[..., :3, 0]))
    R[..., :3, 0] = normalize(torch.cross(R[..., :3, 1], R[..., :3, 2]))
    ray_d = (R @ ray_d[..., None])[..., 0]

    # Compute shading
    ldot = (ray_d * norm).sum(dim=-1).reshape(T)  # T

    def ray_tracing_intersection(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor) -> torch.Tensor:
        # assume all tris batch are the same
        sh = ray_o.shape  # B, S, 3
        tris = tris[:1]  # 1, F, 3, 3
        ray_o = ray_o.view(-1, 3)[None]  # 1, B * S, 3
        ray_d = ray_d.view(-1, 3)[None]  # 1, B * S, 3
        bvh = BVH()  # is this too wasteful, reconstructing the BVH in every iteration?

        # pts: 1, P, 3
        dists_sq, points, face_ids, barys = bvh(tris, ray_o + ray_d * 0.01, ray_d)  # 1, P, 3
        lvis: torch.Tensor = 1 - (dists_sq > 0).float()  # all barycentri coordinates are valid -> intersection -> zero vis
        lvis = lvis.view(*sh[:-1])  # TODO: messy shapes
        lvis.nan_to_num(0.)  # sometimes the moller trumbore returns nan...?
        return lvis

    # Here lren is the indices of the ray direction and pixel to render
    # Perform rendering on lren ray-pixel pair
    if compute_lvis:
        lvis = ray_tracing_intersection(ray_o, ray_d, tris)
    else:
        lvis = torch.ones_like(ldot)
    lvis = lvis.view(B, N, P)
    ldot = ldot.view(B, N, P)
    ray_d = ray_d.view(B, N, P, 3)
    return lvis, ldot, ray_d


def main():
    """
    We have a few assumptions about the ground truth rendering process
    We require a pivot mesh for textures, and other meshes can be loaded with only the vertices
    Since animation should only be about changing the positions of the vertices (without topology warps)

    We don't need to render the full model, only
    1. Normal (geometry only)
    2. Ray-tracing soft shadow (geometry only) (visibility)
    3. Albedo (diffuse albedo map)
    4. Roughness (roughness value map)
    5. Full rendering pipeline? (no, since the material model and indirection illumation is not implemented)

    What do we do?
    """

    # All other related stuff should have been loaded implicitly from the object file's definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'depth', 'surf', 'shade', 'ibl', 'ao'])
    parser.add_argument('--ext', type=str, default='.obj')  # should not change this
    parser.add_argument('--device', type=str, default='cuda')  # should not change this

    # Input output related
    parser.add_argument('--data_root', type=str, default='data/synthetic_human/jody')
    parser.add_argument('--mesh', type=str, default="object/000000.obj")
    parser.add_argument('--width', type=str, default=1024)
    parser.add_argument('--height', type=str, default=1024)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--extri', type=str, default='extri.yml')
    parser.add_argument('--intri', type=str, default='intri.yml')
    parser.add_argument('--output', type=str, default='ray_tracing')

    # Environment map related
    parser.add_argument('--envmap_root', type=str, default='data/lighting/16x32')
    parser.add_argument('--envmap', type=str, default='gym_entrance.hdr')

    # Visualization related
    parser.add_argument('--depth_min', type=float, default=1.0)
    parser.add_argument('--depth_max', type=float, default=5.0)
    parser.add_argument('--surf_min', type=float, default=-3.0)
    parser.add_argument('--surf_max', type=float, default=3.0)
    parser.add_argument('--shading_albedo', type=float, default=0.8)

    # Misc stuff
    # parser.add_argument('--remesh', action='store_true', help='whether to perform remesh before the visibility computation') # slow
    parser.add_argument('--transpose', action='store_true', help='transpose the y and z axis for the synthetic human dataset')
    parser.add_argument('--ground', action='store_true', help='whether the visibility term should consider the ground?')
    parser.add_argument('--sub', nargs='*')

    # Visibility and shading related
    parser.add_argument('--chunk_size', type=int, default=2500, help='chunk size of monte carlo samples (w.r.t. 1 x 512 x 512 image)')
    parser.add_argument('--n_light_sample', type=int, default=50000, help='number of monte carlo samples for each pixel')
    parser.add_argument('--ground_origin', type=float, default=[0, 0, 0], required=False, nargs='*', help='origin of the ground')
    parser.add_argument('--ground_normal', type=float, default=[0, 0, 1], required=False, nargs='*', help='normal of the ground')

    # Prepare arguments
    args = parser.parse_args()
    args.mesh = join(args.data_root, args.mesh)
    args.extri = join(args.data_root, args.extri)
    args.intri = join(args.data_root, args.intri)
    args.output = join(args.data_root, args.output, args.mode)  # {data_root}/ray_tracing/{mode}
    args.envmap = join(args.envmap_root, args.envmap)  # do not merge envmap with data_root
    assert args.ext == '.obj', 'Only obj files are supported'

    # Loading camera intrinsics and extrinsics
    log(f'Loading cameras from {colored(args.intri, "blue")} and {colored(args.extri, "blue")} onto {colored(args.device, "magenta")}')
    camera = read_camera(args.intri, args.extri)  # camera dictionary
    if 'basenames' in camera:
        del camera['basenames']
    for k in list(camera.keys()):
        if args.sub and k not in args.sub:
            del camera[k]  # only render some of the views
    K = torch.from_numpy(np.stack([c['K'] for c in camera.values()]).astype(np.float32)).to(args.device, non_blocking=True)
    R = torch.from_numpy(np.stack([c['R'] for c in camera.values()]).astype(np.float32)).to(args.device, non_blocking=True)
    T = torch.from_numpy(np.stack([c['T'] for c in camera.values()]).astype(np.float32)).to(args.device, non_blocking=True)
    H = int(args.height * args.ratio)
    W = int(args.width * args.ratio)
    K[:, :2] *= args.ratio
    K = get_ndc_perspective_matrix(K, H, W)  # nvdiffrast_K
    B = len(camera)

    # Loading environment map for visibility and shading map computation
    # Shading: visibility term * normal term (normal will provide some local visibility)
    log(f'Loading environment map from {colored(args.envmap, "blue")} onto {colored(args.device, "magenta")}')
    envmap = dotdict()
    envmap.probe = torch.from_numpy(read_hdr(args.envmap)).to(args.device, non_blocking=True)

    # Loading the mesh for rendering, other data loading should be before this
    log(f'Loading meshes from {colored(args.mesh, "blue")} onto {colored(args.device, "magenta")}')
    wverts, faces = load_mesh(args.mesh, args.device)
    log(f'For debugging reason, we\'re only loading the first mesh provided', 'yellow')
    log(f'For now, this script only supports rendering geometry related attributes like normal, shading, soft-shadow', 'yellow')
    if args.transpose:  # Purely for debugging & rendering some datasets quicker
        wverts[..., 0], wverts[..., 1], wverts[..., 2] = wverts[..., 0].clone(), -wverts[..., 2].clone(), wverts[..., 1].clone()  # MARK: weird assignment
    wmesh = Meshes([wverts], [faces])
    wnorm = wmesh.verts_normals_packed()
    attrs = torch.cat([wnorm, wverts], dim=-1)

    # The rasterization (fast, no ray-tracing)
    frame = os.path.splitext(os.path.split(args.mesh)[-1])[0]  # basename as the frame number
    log(f'Rendering {colored(str(B), "magenta")} views for frame {colored(str(frame), "magenta")}')
    attrs, mask = render_nvdiffrast(verts=wverts,
                                    faces=faces,
                                    attrs=attrs,
                                    K=K,
                                    R=R,
                                    T=T,
                                    H=H,
                                    W=W,
                                    R_S=1,
                                    )
    norm, surf = attrs.chunk(2, dim=-1)
    norm = normalize(norm)

    def batch_save_image(img: torch.Tensor, msk: torch.Tensor, output_dir=args.output, ext='.png', action=save_unchanged, dtype=np.uint8):
        log(f'Saving rendered images to {colored(output_dir, "blue")}')
        if msk.ndim == img.ndim - 1:
            msk = msk[..., None]
        img = torch.cat([img, msk], dim=-1)
        parallel_execution([join(output_dir, view, f'{frame}{ext}') for view in camera], [x for x in img.detach().cpu().numpy().astype(dtype)], action=action)

    def normalize_value(value: torch.Tensor, min: float = 0.0, max: float = 1.0, mult=255):
        value = value.clip(min, max)
        value = (value - min) / (max - min)   # perform normalization yourself
        value = value * mult
        return value

    if args.mode == 'normal':
        norm = torch.matmul(R[..., None, None, :, :], norm[..., None])[..., 0]
        norm[..., 1] *= -1
        norm[..., 2] *= -1
        norm = norm * 0.5 + 0.5
        img = normalize_value(norm)
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return

    if args.mode == 'depth':
        C = -torch.matmul(R.mT, T)  # B, 3, 1, camera center in world space
        depth = (surf - C[:, None, None, :, 0]).norm(dim=-1, keepdim=True).expand(-1, -1, -1, 3)
        img = normalize_value(depth, args.depth_min, args.depth_max)
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return

    if args.mode == 'surf':
        img = normalize_value(surf, args.surf_min, args.surf_max)  # perform normalization yourself
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return

    # Now we should compute visibility based on ray triangle intersection
    # Firstly, consider rendering lvis and ldot without ground by ray_tracing
    C = int(args.chunk_size * 512 * 512 / (B * H * W))
    N = args.n_light_sample
    # Faking batch dimensions
    wtris = multi_gather_tris(wverts, faces)[None]  # this is some heavy operation
    valid = mask.nonzero(as_tuple=True)
    surf_valid = surf[valid][None]
    norm_valid = norm[valid][None]

    ao = 0
    shade = 0
    for i in tqdm(range(0, N, C)):
        CC = min(N, i + C) - i  # the remaining chunk size (C or smaller)
        lvis_chunk, ldot_chunk, ray_d_chunk = light_visibility(surf_valid,
                                                               norm_valid,
                                                               wtris,
                                                               CC,
                                                               args.mode != 'ibl')  # B, N, P
        light = sample_envmap_image(envmap.probe, ray_d_chunk)  # B, N, P
        shade += (lvis_chunk[..., None] * ldot_chunk[..., None] * light).sum(dim=-3)  # B, P,
        ao += lvis_chunk.sum(dim=-2)  # B, P,

    # Normalization of final values
    ao = ao / N
    shade = shade * 2.0 * np.pi / N / np.pi * args.shading_albedo  # only upper sphere are sampled, brdf sum to 1
    # shade = linear2srgb(shade)

    # Store back extracted values
    ao = torch.zeros_like(mask, dtype=torch.float).index_put_(valid, ao[0])
    shade = torch.zeros_like(surf, dtype=torch.float).index_put_(valid, shade[0])

    if args.mode == 'ao':
        ao = ao.view(B, H, W)[..., None].expand(-1, -1, -1, 3)  # B, H, W, 3
        img = normalize_value(ao, max=512)  # 0 - 1?
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return

    if args.mode == 'shade':
        shade = shade.view(B, H, W, 3)  # B, H, W, 3
        img = normalize_value(shade)  # 0 - 1?
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return

    if args.mode == 'ibl':
        shade = shade.view(B, H, W, 3)  # B, H, W, 3
        img = normalize_value(shade)  # 0 - 1?
        mask = normalize_value(mask)
        batch_save_image(img, mask)
        return


if __name__ == '__main__':
    main()
