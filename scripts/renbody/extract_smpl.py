import os
import cv2
import torch
import argparse
import numpy as np
import trimesh
from glob import glob
from os.path import join, exists, dirname

from easymocap.mytools.file_utils import read_json
from easyvolcap.utils.easy_utils import load_bodymodel
from easyvolcap.utils.console_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/renbody')
    parser.add_argument('--smpl-dir', type=str, default='output/smpl')
    parser.add_argument('--scenes', type=str, nargs='+', default=['0021_08'])
    args = parser.parse_args()
    return args


@catch_throw
def main(args):
    for scene in tqdm(args.scenes):
        motion_files = glob(join(args.input, scene, args.smpl_dir, '*.json'))
        motion_files.sort()
        check_data = read_json(motion_files[0])
        if isinstance(check_data, dict):
            motion = [read_json(x)['annots'][0] for x in motion_files]
        else:
            motion = [read_json(x)[0] for x in motion_files]
        motion_dict = {}
        for key in motion[0].keys():
            if key not in motion_dict:
                motion_dict[key] = []
            for i in range(len(motion)):
                motion_dict[key].append(np.array(motion[i][key], dtype=np.float32))
        motion_dict.pop('id')
        for key in motion_dict.keys():
            motion_dict[key] = np.concatenate(motion_dict[key], axis=0)    
        np.savez(join(args.input, scene, 'motion.npz'), **motion_dict)
        model = load_bodymodel(join(args.input, scene), join(args.smpl_dir, '../cfg_exp.yml'))
        verts = model(return_tensor=False, shapes=motion_dict['shapes'], poses=motion_dict['poses'], Rh=motion_dict['Rh'], Th=motion_dict['Th'])
        output_dir = join(args.input, scene, 'smpl/points')
        os.makedirs(output_dir, exist_ok=True)
        for i in range(verts.shape[0]):
            vert = verts[i]
            mesh = trimesh.Trimesh(vertices=vert, faces=model.faces)
            mesh.export(join(output_dir, f'{i:06d}.ply'))
        

if __name__ == '__main__':
    args = parse_args()
    main(args)