"""
Resample point clouds from one folder to another
"""
import torch
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts, export_pts, to_tensor, to_cuda
from easyvolcap.utils.fcds_utils import random, farthest, surface_points, voxel_surface_down_sample, voxel_down_sample, SamplingType


@catch_throw
def main():
    args = dotdict(
        input='data/iphone_stage/evc_0319-10/pcds',
        output='data/iphone_stage/evc_0319-10/pcds6k',
        n_points=6000,  # downsample or upsample to this number of points,
        voxel_size=0.005,
        type=SamplingType.RANDOM_DOWN_SAMPLE.name,
        device='cuda',
        color=0.0,
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    args.type = SamplingType[args.type]

    pcds = sorted(os.listdir(args.input))
    outs = [f.replace(args.input, args.output) for f in pcds]

    for i, (pcd, out) in enumerate(zip(tqdm(pcds), outs)):
        pcd = join(args.input, pcd)
        out = join(args.output, out)
        # TODO: Add support for preserving vertex properties
        xyz = to_cuda(load_pts(pcd)[0], device=args.device)[None]  # only need xyz for now
        if args.type == SamplingType.RANDOM_DOWN_SAMPLE:
            xyz = random(xyz, n_points=args.n_points, std=0.0)  # no extra perturbation
        elif args.type == SamplingType.FARTHEST_DOWN_SAMPLE:
            xyz = farthest(xyz, n_points=args.n_points)
        elif args.type == SamplingType.VOXEL_DOWN_SAMPLE:
            xyz = voxel_down_sample(xyz, voxel_size=args.voxel_size)
        elif args.type == SamplingType.SURFACE_DISTRIBUTION:
            xyz = surface_points(xyz, n_points=args.n_points)
        elif args.type == SamplingType.MARCHING_CUBES_RECONSTRUCTION:
            xyz = voxel_surface_down_sample(xyz, n_points=args.n_points, voxel_size=args.voxel_size)
        else:
            raise NotImplementedError
        if args.color >= 0:
            rgb = torch.full_like(xyz, args.color)
        else:
            rgb = torch.rand_like(xyz)
        export_pts(xyz, rgb, filename=out)


if __name__ == '__main__':
    main()
