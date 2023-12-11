# Save a pytorch module's runtime buffers and parameters to an npz file, since numpy supports are more portable than pytorch's (and python's pickle)
# Note that we're saving the "runtime" tensors, so that the target platform can perform inference without worrying about dataloading, etc.
# This function will try to invoke evc programmatically
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy


@catch_throw
def main():
    # fmt: off
    import sys
    sys.path.append('.')

    sep_ind = sys.argv.index('--')
    our_args = sys.argv[1:sep_ind]
    evv_args = sys.argv[sep_ind + 1:]
    sys.argv = [sys.argv[0]] + ['-t','test'] + evv_args

    # Entry point first, other modules later to avoid strange import errors
    from easyvolcap.engine import cfg
    from easyvolcap.scripts.main import test # will do everything a normal user would do
    from easyvolcap.runners.volumetric_video_runner import VolumetricVideoRunner
    # fmt: on

    # Actual construction and loading of the network
    runner: VolumetricVideoRunner = test(cfg, dry_run=True)
    epoch = runner.load_network()  # will perform necessary dumping and preprocessing under the hood
    runner.model.eval()

    # Save the model's registered parameters as numpy arrays in npz
    state_dict = runner.model.state_dict()
    param_dict = to_numpy(state_dict)  # a shallow dict
    log('Will save these keys to dict: ', param_dict.keys())
    npz_path = join(runner.trained_model_dir, f'{epoch - 1}.npz')
    np.savez_compressed(npz_path, **param_dict)
    log(f'Compressed numpy archive save to {blue(npz_path)}')


if __name__ == '__main__':
    main()
