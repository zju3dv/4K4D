"""
Load the pretrained model of ENeRF's IBR module
Will serve as a blender for image based rendering of the colors
How do we determine which view to use for the blending?
Or just use median value?
This seems counter-intuitive
Assuming shared camera parameters for now

How to get the correct source views?
Assuming you've got a good position, we unproject all points on all views, then perform argsort on depth
For every point, then find the closest ranking view, and use that view's color
This way we can deal with non-uniform camera distribution
"""

import torch
import torch.nn.functional as F
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.viewer_utils import Camera
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.easy_utils import read_camera, write_camera
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter
from easyvolcap.utils.data_utils import load_pts, export_pts, to_cuda, load_image
from easyvolcap.utils.image_utils import fill_nhwc_image
from easyvolcap.utils.ibr_utils import sample_geometry_feature_image


@catch_throw
def main():
    # Prepare arguments
    args = dotdict(
        data_root='data/mobile_stage/dance3',
        images_dir='images',
        cameras_dir='optimized',
        input='surfs6k',
        output='surfs6k',
        n_srcs=3,
        ratio=0.25,  # smaller ratio for blurrier points
        sequential_image_loading=False,

        # TODO: Use IBR models for this, for now, they require voxel feature input thus cannot be easily separated
        # enerfi_path='data/trained_model/enerfi_dtu',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))

    # # Prepare ENeRF IBR model
    # enerfi = load_pretrained(args.enerfi_path)
    # breakpoint()

    # Perform depth ranking sorting
    cameras = read_camera(join(args.data_root, args.cameras_dir))  # cameras sorted by camera names
    camera_names = sorted(cameras)
    cameras = dotdict({cam: cameras[cam] for cam in camera_names})
    batches = dotdict({cam: to_cuda(Camera().from_easymocap(cameras[cam]).to_batch()) for cam in camera_names})
    Rs = torch.stack([batches[cam].R for cam in batches])  # V, 3, 3
    Ts = torch.stack([batches[cam].T for cam in batches])  # V, 3, 1
    Ks = torch.stack([batches[cam].K for cam in batches])  # V, 3, 3
    Ks[..., :2, :] *= args.ratio
    src_ixts = Ks  # V, 3, 3
    src_exts = torch.cat([Rs, Ts], dim=-1)  # V, 3, 4

    files = sorted(os.listdir(join(args.data_root, args.input)))
    images = sorted(os.listdir(join(args.data_root, args.images_dir, camera_names[0])))
    for file in tqdm(files):
        # Perform dataloading and transfer
        idx = int(splitext(file)[0])
        xyz = load_pts(join(args.data_root, args.input, file))[0]
        xyz = to_cuda(xyz)  # N, 3

        # Load all images for this frame
        log(f'Loading images for frame {idx}')
        imgs = parallel_execution([join(args.data_root, args.images_dir, cam, images[idx]) for cam in camera_names], ratio=args.ratio, action=load_image, sequential=args.sequential_image_loading)
        imgs = to_cuda(imgs)  # V, Hx, Wx, 3
        Hs = torch.as_tensor([img.shape[-3] for img in imgs], device=xyz.device)  # V,
        Ws = torch.as_tensor([img.shape[-2] for img in imgs], device=xyz.device)  # V,
        mh, mw = max(Hs), max(Ws)
        imgs = [fill_nhwc_image(img, [mh, mw]) for img in imgs]
        imgs = torch.stack(imgs).permute(0, 3, 1, 2)  # V, H, W, 3 -> V, 3, H, W

        # Perform the actual depth ranking depth.argsort() should have the same value as ranking.argsort()
        log(f'Depth ranking for frame {idx}')
        depth = (xyz[None] @ Rs.mT + Ts.mT)[..., -1]  # V, N
        scr = (xyz[None] @ Rs.mT + Ts.mT) @ Ks.mT  # V, N, 3
        scr = scr[..., :2] / scr[..., 2:]  # V, N, 2
        depth = torch.where((scr[..., 0] < 0) | (scr[..., 0] > Ws[..., None]) | (scr[..., 1] < 0) | (scr[..., 1] > Hs[..., None]), torch.inf, depth)
        argsort = depth.argsort(dim=-1)  # V, N
        rankings = torch.empty_like(argsort).scatter_(dim=-1, index=argsort, src=torch.arange(argsort.shape[-1], device=xyz.device)[None].repeat(len(depth), 1))

        # Find the best k source views
        src_inds = rankings.topk(args.n_srcs, dim=0, largest=False).indices  # S, N
        src_inds = src_inds.mT  # N, S

        # Sample RGB color from them
        log(f'Sampling rgb for frame {idx}')
        rgbs = sample_geometry_feature_image(xyz[None], imgs[None], src_exts[None], src_ixts[None], torch.ones(2, 1, device=xyz.device))[0]  # V, N, 3
        rgbs = multi_gather(rgbs.permute(1, 0, 2), src_inds, dim=-2)  # N, S, 3
        rgb = rgbs.mean(dim=-2)

        # Save final output to file
        export_pts(xyz, rgb, filename=join(args.data_root, args.output, file))


if __name__ == '__main__':
    main()
