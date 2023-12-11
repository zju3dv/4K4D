# The configuration system used in this project: yacs (yaml based)
# does not provide a good enough support for loading multiple parent configuration files
# however, the baseline experiments might result in bloated configs
# we need two different kinds of config source: data and experiment

import os
import ruamel.yaml as yaml

from glob import glob
from os.path import join
import argparse


def walk_config(exp, data, exp_name, data_name, exp_keys):
    for key in exp_keys:
        if key in exp and key in data:
            if isinstance(exp[key], dict) and isinstance(data[key], dict):
                walk_config(exp[key], data[key], exp_name, data_name, exp_keys)
            elif isinstance(exp[key], str) and isinstance(data[key], str):
                data[key] = exp[key].replace(exp_name, data_name)
            else:
                raise NotImplementedError('Unsupported config type to replace')


def main():

    exp_keys = [
        'relighting_cfg',
        'exp_name',
        'parent_cfg',
        'geometry_mesh',
        'geometry_pretrain',
    ]  # other keys are data related keys (shared across experiments)
    datasets = ['mobile_stage', 'synthetic_human']
    experiments = ['nerf', 'neuralbody', 'brute']
    data_file_prefix = 'base'  # we define data entries here
    exp_file_template = 'configs/synthetic_human/base_synthetic_jody.yaml'  # this means we've defined exp entries on jody
    configs_root = 'configs'

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_keys', nargs='+', default=exp_keys)
    parser.add_argument('--experiments', nargs='+', default=experiments)
    parser.add_argument('--datasets', nargs='+', default=datasets)
    parser.add_argument('--data_file_prefix', type=str, default=data_file_prefix)
    parser.add_argument('--exp_file_template', type=str, default=exp_file_template)
    parser.add_argument('--configs_root', type=str, default=configs_root)
    args = parser.parse_args()
    exp_keys = args.exp_keys
    datasets = args.datasets
    experiments = args.experiments
    data_file_prefix = args.data_file_prefix
    exp_file_template = args.exp_file_template
    configs_root = args.configs_root

    for dataset in datasets:
        data_files = glob(join(configs_root, dataset, f'{data_file_prefix}*'))
        for experiment in experiments:
            exp_files = [exp_file_template.replace(data_file_prefix, experiment) for f in data_files]
            for data_file, exp_file in zip(data_files, exp_files):
                exp_name = os.path.splitext(exp_file)[0].split('_')  # something like synthetic_jody
                exp_name = '_'.join(exp_name[-2:])
                data_name = os.path.splitext(data_file)[0].split('_')  # something like jody / josh
                data_name = '_'.join(data_name[-2:])
                out_file = data_file.replace(data_file_prefix, experiment)

                exp = yaml.round_trip_load(open(exp_file))
                data = yaml.round_trip_load(open(data_file))

                # inplace modification
                walk_config(exp, data, exp_name, data_name, exp_keys)
                yaml.round_trip_dump(data, open(out_file, 'w'))


if __name__ == '__main__':
    main()
