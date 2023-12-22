# Given input point cloud locations, sample rgbras

# This function will try to invoke evc programmatically
import torch
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy, export_pts, load_pts


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t','test'] + evv_args

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pcd', type=str, default='data/NHR/sport_1_easymocap/vhulls_backup/000000.ply')
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--sampler', type=str, default='TurboChargedR4DV')
    parser.add_argument('--sub_sampler', type=str, default='TurboChargedR4DV')
    parser.add_argument('--exp_name', type=str, default='tcr4dv_sport1_subsample')
    args = parser.parse_args(our_args)

    sys.argv += [f'val_dataloader_cfg.dataset_cfg.frame_sample={args.frame_idx},{args.frame_idx + 1},1']

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.engine import cfg
    from easyvolcap.engine import SAMPLERS
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    epoch = runner.load_network()
    runner.model.eval()

    if 'fg_sampler_cfg' in cfg.model_cfg.sampler_cfg: cfg.model_cfg.sampler_cfg.fg_sampler_cfg.type = args.sub_sampler
    if 'bg_sampler_cfg' in cfg.model_cfg.sampler_cfg: cfg.model_cfg.sampler_cfg.bg_sampler_cfg.type = args.sub_sampler
    cfg.model_cfg.sampler_cfg.type = args.sampler
    sampler = SAMPLERS.build(cfg.model_cfg.sampler_cfg, network=runner.model.network)

    pcd, _, _, _ = load_pts(args.input_pcd)  # we only care about the point locations
    runner.model.sampler.pcds[0] = torch.as_tensor(pcd).to(runner.model.sampler.pcds[0]) # FIXME: index mismatch

    # Save a pytorch loadable model to disk for actual visualization
    sampler.construct_from_runner(runner)
    runner.model.sampler = sampler
    runner.trained_model_dir = f'data/trained_model/{args.exp_name}'
    runner.optimizer = None  # only this should not have been saved
    runner.save_model(epoch - 1, latest=False)


if __name__ == '__main__':
    main()
