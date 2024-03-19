"""
Resample point clouds from one folder to another
"""
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_pts, export_pts, to_tensor, to_cuda
from easyvolcap.utils.fcds_utils import random, farthest, surface_points, voxel_surface_down_sample, voxel_down_sample, SamplingType


@catch_throw
def main():
    args = dotdict(
        input='data/actorshq/Actor01/Sequence1/1x/surfs',
        output='data/actorshq/Actor01/Sequence1/1x/surfs36k',
        n_points=36000,  # downsample or upsample to this number of points,
        type=SamplingType.RANDOM_DOWN_SAMPLE.name,
        device='cuda',
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
            xyz = random(xyz, n_points=args.n_points, std=0.0)
        else:
            raise NotImplementedError
        export_pts(xyz, filename=out)


if __name__ == '__main__':
    main()
