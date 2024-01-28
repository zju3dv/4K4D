# Save a pytorch module's runtime buffers and parameters to an npz file, since numpy supports are more portable than pytorch's (and python's pickle)
# Note that we're saving the "runtime" tensors, so that the target platform can perform inference without worrying about dataloading, etc.
# This function will try to invoke evc programmatically
import torch
import argparse
import numpy as np

from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.fcds_utils import voxel_down_sample, remove_outlier
from easyvolcap.utils.data_utils import to_numpy, to_x, export_pts, export_mesh, to_tensor
from easyvolcap.utils.math_utils import point_padding, affine_padding, affine_inverse
from easyvolcap.utils.chunk_utils import multi_gather, multi_scatter


@catch_throw
def main():
    # fmt: off
    import sys

    sep_ind = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t','test'] + evv_args

    args = dotdict()
    args.result_dir = 'data/geometry'
    args.frame_index = 0
    args.skip_align = False
    args =dotdict(vars(build_parser(args).parse_args(our_args)))

    sys.argv += [f'val_dataloader_cfg.dataset_cfg.frame_sample={args.frame_index},{args.frame_index+1},1']

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.engine import cfg
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    # Actual construction and loading of the network
    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    dataset = runner.val_dataloader.dataset
    epoch = runner.load_network()  # will perform necessary dumping and preprocessing under the hood
    runner.model.eval()

    special_mapping = {
        f'sampler.pcds.{args.frame_index}': 'pts',
        f'sampler.rgbs.{args.frame_index}': 'color',
        f'sampler.bg_sampler.pcds.{args.frame_index}': 'pts',
        f'sampler.bg_sampler.rgbs.{args.frame_index}': 'color',
        f'sampler.fg_sampler.pcds.{args.frame_index}': 'pts',
        f'sampler.fg_sampler.rgbs.{args.frame_index}': 'color',
    }

    named_mapping = {
        f'sampler.rads.{args.frame_index}': 'radius',
        f'sampler.occs.{args.frame_index}': 'alpha',
        f'sampler.bg_sampler.rads.{args.frame_index}': 'radius',
        f'sampler.bg_sampler.occs.{args.frame_index}': 'alpha',
        f'sampler.fg_sampler.rads.{args.frame_index}': 'radius',
        f'sampler.fg_sampler.occs.{args.frame_index}': 'alpha',
    }

    # Save the model's registered parameters as numpy arrays in npz
    state_dict = runner.model.state_dict()
    numpy_dict = to_numpy(to_x(state_dict, torch.float))  # a shallow dict
    output = dotdict()
    output.scalars = dotdict()
    for k, v in numpy_dict.items():
        if k in special_mapping:
            output[special_mapping[k]] = v  # only the first
        elif k in named_mapping:
            output.scalars[named_mapping[k]] = v
        else:
            log(yellow(f'{k} will be discarded'))

    log('Will save these keys: ', output.keys())
    ply_path = join(args.result_dir, cfg.exp_name, 'SPLATS', f'frame{args.frame_index:04d}.ply')
    log(f'Point cloud and scalars save to {blue(ply_path)}')

    if dataset.use_aligned_cameras and not args.skip_align:  # match the visual hull implementation
        mat = affine_padding(dataset.c2w_avg)  # 4, 4
        output.pts = (point_padding(to_tensor(output.pts)) @ mat.mT)[..., :3]  # homo
        output.pts = to_numpy(output.pts)

    export_pts(**output, filename=ply_path)


if __name__ == '__main__':
    main()
