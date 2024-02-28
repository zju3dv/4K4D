"""
Load marigold args.depth
Load easyvolcap cameras
Produce a back-projected and merged point cloud
"""

import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.ray_utils import get_rays
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.cam_utils import compute_camera_similarity
from easyvolcap.utils.image_utils import resize_image, pad_image
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.math_utils import affine_padding, affine_inverse
from easyvolcap.utils.data_utils import to_cuda, to_tensor, to_x, add_batch, load_image, load_depth, export_pts, read_pfm
from easyvolcap.utils.fusion_utils import filter_global_points, depth_geometry_consistency, compute_consistency


@catch_throw
def main():
    args = dotdict()
    args.data_root = 'data/renbody/0008_01'
    args.depth_dir = 'depths'
    args.depth = '000000.exr'  # camera + postfix = args.depth file name
    args.images_dir = 'images_calib'
    args.image = '000000.jpg'
    args.cameras_dir = 'optimized'
    args.output = f'{args.image.replace(".jpg", ".ply")}'
    args.n_srcs = 4
    args.ratio = 0.5
    args.scale = 1.0
    args.geo_abs_thresh = 1.0
    args.geo_rel_thresh = 0.01
    args.skip_depth_consistency = False
    args = dotdict(vars(build_parser(args).parse_args()))

    cameras = to_tensor(read_camera(join(args.data_root, args.cameras_dir)))
    # nv = min(len(cameras), len(os.listdir(join(args.data_root, args.depth_dir))))
    # cameras = dotdict({k: v for k in sorted(cameras)[:nv]})
    names = sorted(os.listdir(join(args.data_root, args.depth_dir)))
    cameras = dotdict({k: cameras[k] for k in names})

    w2cs = torch.stack([torch.cat([cameras[k].R, cameras[k].T], dim=-1) for k in cameras])  # V, 4, 4
    c2ws = affine_inverse(w2cs)
    _, src_inds = compute_camera_similarity(c2ws, c2ws)  # V, V

    dpts = []
    cens = []
    dirs = []

    rgbs = []
    Ks = []

    for cam in tqdm(cameras, desc='Loading depths & images & rays'):
        depth_file = join(args.data_root, args.depth_dir, cam, args.depth)
        image_file = join(args.data_root, args.images_dir, cam, args.image)
        rgb = to_cuda(to_tensor(load_image(image_file)).float())  # H, W, 3
        dpt = to_cuda(to_tensor(load_depth(depth_file)).float()) * args.scale  # H, W, 1

        H, W = cameras[cam].H, cameras[cam].W
        K, R, T = cameras[cam].K, cameras[cam].R, cameras[cam].T
        K, R, T = to_x(to_cuda([K, R, T]), torch.float)

        K[0:1] *= int(W * args.ratio) / W
        K[1:2] *= int(H * args.ratio) / H
        H, W = int(H * args.ratio), int(W * args.ratio)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = resize_image(rgb, size=(H, W))
        if dpt.shape[0] != H or dpt.shape[1] != W:
            dpt = resize_image(dpt, size=(H, W))

        ray_o, ray_d = get_rays(H, W, K, R, T, z_depth=True)
        dpts.append(dpt)
        cens.append(ray_o)
        dirs.append(ray_d)
        rgbs.append(rgb)
        Ks.append(K)

    H, W = max([d.shape[-3] for d in dpts]), max([d.shape[-2] for d in dpts])

    dpts = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in dpts])  # V, H, W, 1
    cens = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in cens])  # V, H, W, 3
    dirs = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in dirs])  # V, H, W, 3
    rgbs = torch.stack([pad_image(i.permute(2, 0, 1), (H, W)).permute(1, 2, 0) for i in rgbs])  # V, H, W, 3
    Ks = torch.stack(Ks)  # V, 3, 3

    xyzs_out = []
    rgbs_out = []

    if not args.skip_depth_consistency:

        # Perform args.depth consistency check and filtering
        for v in tqdm(range(len(cameras)), desc='Fusing depths'):
            # Prepare source views' information
            src_ind = src_inds[v, 1:1 + args.n_srcs]  # 4,
            dpt_src = dpts[src_ind]  # 4, HW
            ixt_src = Ks[src_ind]  # 4, 3, 3
            ext_src = affine_padding(w2cs[src_ind])  # 4, 3, 3

            # Prepare reference view's information
            dpt_ref = dpts[v]  # HW, 1
            ixt_ref = Ks[v]  # 3, 3
            ext_ref = affine_padding(w2cs[v])  # 4, 4

            # Prepare data for computation
            S, H, W, C = dpt_src.shape
            dpt_src = dpt_src.view(S, H, W)  # 4, H, W
            dpt_ref = dpt_ref.view(H, W)
            ixt_ref, ext_ref, ixt_src, ext_src = to_x(to_cuda([ixt_ref, ext_ref, ixt_src, ext_src]), torch.float)

            depth_est_averaged, photo_mask, geo_mask, final_mask = compute_consistency(
                dpt_ref, ixt_ref, ext_ref, dpt_src, ixt_src, ext_src,
                args.geo_abs_thresh, args.geo_rel_thresh
            )

            # Filter points based on geometry and photometric mask
            ind = final_mask.view(-1).nonzero()  # N, 1 # MARK: SYNC
            dpt = multi_gather(depth_est_averaged.view(-1, 1), ind)  # N, 1
            dir = multi_gather(dirs[v].view(-1, 3), ind)  # N, 3
            cen = multi_gather(cens[v].view(-1, 3), ind)  # N, 3
            rgb = multi_gather(rgbs[v].view(-1, 3), ind)  # N, 3
            xyz = cen + dpt * dir  # N, 3
            xyzs_out.append(xyz)
            rgbs_out.append(rgb)

            log(f'View {v}, photo_mask {photo_mask.sum() / photo_mask.numel():.04f}, geometry mask {geo_mask.sum() / geo_mask.numel():.04f}, final mask {final_mask.sum() / final_mask.numel():.04f}, final point count {len(xyz)}')

    else:
        xyzs_out = [(cens[v] + dpts[v] * dirs[v]).view(-1, 3) for v in range(len(cameras))]
        rgbs_out = [rgb.view(-1, 3) for rgb in rgbs]

    xyz = torch.cat(xyzs_out, dim=-2)
    rgb = torch.cat(rgbs_out, dim=-2)
    filename = join(args.data_root, args.output)
    export_pts(xyz, rgb, filename=filename)
    log(yellow(f'Fused points saved to {blue(filename)}, totally {cyan(xyz.numel() // 3)} points'))


if __name__ == '__main__':
    main()
