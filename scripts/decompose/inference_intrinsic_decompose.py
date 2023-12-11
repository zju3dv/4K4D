# need to install: https://github.com/dendenxu/intrinsic-image-decomposition
# it's a private fork from: https://github.com/DefUs3r/Intrinsic-Image-Decomposition

import os
import argparse
from tqdm import tqdm
from os.path import join
from termcolor import colored

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.console_utils import log, run
from easyvolcap.utils.parallel_utils import parallel_execution
# fmt: on


def decompose_one_image(input_path: str, albedo_output_path: str, shading_output_path: str):
    os.makedirs(os.path.dirname(albedo_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(shading_output_path), exist_ok=True)

    os.chdir(intrinsic_image_decomposition)
    run(f'python decompose.py {input_path} -r {albedo_output_path} -s {shading_output_path}', quite=True)

parser = argparse.ArgumentParser()
parser.add_argument('--code_root', default='/home/xuzhen/code/intrinsic-image-decomposition')
parser.add_argument('--data_root', default='data/my_zjumocap/my_313')
parser.add_argument('--image_dir', default='images')
parser.add_argument('--albedo_dir', default='albedo')
parser.add_argument('--shading_dir', default='shading')
parser.add_argument('--frame_sync', action='store_false', dest='view_sync')
args = parser.parse_args()

intrinsic_image_decomposition = args.code_root
data_root = args.data_root
image_dir = args.image_dir
albedo_dir = args.albedo_dir
shading_dir = args.shading_dir
view_sync = args.view_sync  # will batch through all views instead of a single camera

image_path = join(data_root, image_dir)
views = sorted(os.listdir(image_path))
frames = sorted(os.listdir(join(image_path, os.listdir(image_path)[0])))

iteration = frames if view_sync else views
inputs = [[join(image_path, v, f) for v in views] for f in frames] if view_sync else [[join(image_path, v, f) for f in frames] for v in views]
albedos = [[v.replace(image_dir, albedo_dir) for v in f] for f in inputs]
shadings = [[v.replace(image_dir, shading_dir) for v in f] for f in inputs]

inputs = [[os.path.abspath(v) for v in f] for f in inputs]  
albedos = [[os.path.abspath(v) for v in f] for f in albedos]  
shadings = [[os.path.abspath(v) for v in f] for f in shadings]  

for i in tqdm(range(len(iteration))):
    log(f'processing {iteration[i]}: {colored(f"#{i+1}/{len(iteration)}", "magenta")}')
    parallel_execution(inputs[i], albedos[i], shadings[i], action=decompose_one_image)
