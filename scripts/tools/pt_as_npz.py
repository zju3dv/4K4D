# Save a pytorch module's runtime buffers and parameters to an npz file, since numpy supports are more portable than pytorch's (and python's pickle)
# Note that we're saving the "runtime" tensors, so that the target platform can perform inference without worrying about dataloading, etc.
# This function will try to invoke evc programmatically
import os
import torch
import argparse
import numpy as np
from os.path import join
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import to_numpy


@catch_throw
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/trained_model/enerf_dtu/latest.pt')
    parser.add_argument('--model_only', action='store_true')
    args = parser.parse_args()

    # Save the model's registered parameters as numpy arrays in npz
    state_dict = torch.load(args.input, map_location='cpu')
    if args.model_only:
        state_dict = state_dict['model']
    param_dict = to_numpy(state_dict)  # a shallow dict
    log('Will save these keys to dict: ', param_dict.keys())
    npz_path = args.input.replace('.pt', '.npz')
    np.savez_compressed(npz_path, **param_dict)
    log(f'Compressed numpy archive save to {blue(npz_path)}')


if __name__ == '__main__':
    main()
