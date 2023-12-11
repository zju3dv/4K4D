# Prepares weight for finetuning, will copy corresponding weights from pretrained model
# Mark them as fresh (epoch -1) (will start running with epoch 0)
# Then remember the pretrained epoch as a `pretrain_epoch` attribute in the checkpoint

import os
import torch
import argparse
from os.path import join, dirname
from easyvolcap.utils.console_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pretrain', default='enerf_dtu')
    parser.add_argument('finetune', default='enerf_my_313_se')
    parser.add_argument('--trained_model_dir', default='data/trained_model')
    parser.add_argument('--trained_model_file', default='latest.pt')
    args = parser.parse_args()

    pretrain_path = join(args.trained_model_dir, args.pretrain, 'latest.pt')
    finetune_path = join(args.trained_model_dir, args.finetune, 'latest.pt')

    model = torch.load(pretrain_path, map_location='cpu')
    model['pretrain_epoch'] = model['epoch']
    model['epoch'] = -1  # will start as 0

    os.makedirs(dirname(finetune_path), exist_ok=True)
    torch.save(model, finetune_path)

    log(f'{blue(pretrain_path)} (epoch: {model["pretrain_epoch"]}) -> {blue(finetune_path)} (epoch: {model["epoch"]})')


if __name__ == '__main__':
    main()
