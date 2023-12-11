import os
import cv2
import torch
import argparse
import numpy as np
from os.path import join, exists, dirname

import smplx
import trimesh

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.data_utils import load_dotdict, to_tensor

models = {
    'smplx': smplx.create(
        './data/models',
        model_type='smplx',
        num_betas=10,
        use_pca=False,
        ext='pkl'
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/renbody')
    parser.add_argument('--scenes', type=str, nargs='+', default=['0013_06'])
    parser.add_argument('--model_type', type=str, default='smplx')
    args = parser.parse_args()
    return args

@catch_throw
def main(args):
    model = models[args.model_type]
    for scene in tqdm(args.scenes):
        output_dir = join(args.input, scene, 'smplx/points')
        os.makedirs(output_dir, exist_ok=True)
        motion = to_tensor(load_dotdict(join(args.input, scene, 'motion.npz')))
        out = model(**motion)
        for i in range(len(out.vertices)):
            mesh = trimesh.Trimesh(vertices=out.vertices[i], faces=model.faces)
            mesh.export(join(output_dir, f'{i:06d}.ply'))
        

if __name__ == '__main__':
    args = parse_args()
    main(args)