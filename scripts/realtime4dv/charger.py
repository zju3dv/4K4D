# This function will try to invoke evc programmatically
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy
from easyvolcap.utils.net_utils import save_npz


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t','test'] + evv_args

    import torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampler', type=str, default='SuperChargedR4DVB')
    parser.add_argument('--sub_sampler', type=str, default='SuperChargedR4DV')
    parser.add_argument('--exp_name', type=str, default='scr4dvb_dance3')
    parser.add_argument('--save_fp32', action='store_true')
    parser.add_argument('--save_pt', action='store_true')
    parser.add_argument('--no_save_npz', action='store_false', dest='save_npz')
    args = parser.parse_args(our_args)

    # You have to save at least one type of model
    assert args.save_pt or args.save_npz

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
    sampler = SAMPLERS.build(cfg.model_cfg.sampler_cfg, network=runner.model.network, dtype=torch.float if args.save_fp32 else torch.half)

    # Save a pytorch loadable model to disk for actual visualization
    sampler.construct_from_runner(runner)
    runner.model.sampler = sampler
    runner.trained_model_dir = f'data/trained_model/{args.exp_name}'
    runner.optimizer = None  # only this should not have been saved

    if args.save_pt:
        runner.save_model(epoch - 1, latest=False)

    if args.save_npz:
        # Save the model's optimizable parameters as numpy arrays in npz
        # state_dict = runner.model.state_dict()
        # param_dict = to_numpy(state_dict)  # a shallow dict
        # log('Will save these keys to dict: ', param_dict.keys())
        # npz_path = join(runner.trained_model_dir, f'{epoch - 1}.npz')
        # os.makedirs(dirname(npz_path), exist_ok=True)
        # np.savez_compressed(npz_path, **param_dict)
        # log(f'Compressed numpy archive save to {blue(npz_path)}')
        save_npz(runner.model, runner.trained_model_dir, epoch - 1)


if __name__ == '__main__':
    main()
