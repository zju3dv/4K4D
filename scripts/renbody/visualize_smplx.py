import os
import cv2
from PIL import Image
import torch
import argparse
import numpy as np
import trimesh
from os.path import join, exists, dirname

from easyvolcap.utils.console_utils import *
from easymocap.mytools.camera_utils import read_cameras
from easymocap.visualize.render_func import get_render_func


def generate_video(result_str: str,
                   fps: int = 30,
                   crf: int = 17,
                   vid_ext: str = '.mp4',
                   vcodec: str = 'libx264',
                   pix_fmt: str = 'yuv420p',  # chrome friendly
                   ):
    output = result_str.split('*')[0][:-1] + vid_ext  # remove ", remove *, remove / and remove -"
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-framerate', fps,
        '-f', 'image2',
        '-pattern_type', 'glob',
        '-nostdin',  # otherwise you cannot chain commands together
        '-y',
        '-r', fps,
        '-i', f'"{result_str}"',
        '-c:v', vcodec,
        '-crf', crf,
        '-pix_fmt', pix_fmt,
        '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',  # avoid yuv420p odd number bug
        output,
    ]
    run(cmd)
    return output


def main(args):
    backend = 'pyrender'
    render_func = get_render_func('image', backend)
    for scene in args.scenes: 
        cameras = read_cameras(join(args.input, scene))
        smplx_dir = join(args.input, scene, f'{args.model}/points')
        smplx_paths = os.listdir(smplx_dir)
        smplx_paths.sort()
        if len(args.subs) > 0:
            keys = args.subs
        else:
            keys = cameras.keys()
        for smplx_path in tqdm(smplx_paths):
            frame = os.path.basename(smplx_path).split('.')[0]
            mesh = trimesh.load(join(smplx_dir, smplx_path))
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
            data = {0: {'vertices': vertices, 'faces': faces, 'name': '0'}}
            images = {}
            cameras_new = {}
            for key in keys:
                image = Image.open(join(args.input, scene, 'images', key, f'{frame}.jpg'))
                images[key] = np.array(image)
                cameras_new[key] = cameras[key]
            output = render_func(images, data, cameras_new, {})
            for key in keys:
                output_dir = join(args.input, scene, f'{args.model}/render/{key}')
                os.makedirs(output_dir, exist_ok=True)
                outname = join(output_dir, f'{frame}.jpg')
                image = Image.fromarray(output[key])
                image.save(outname)
        if args.video:
            for key in keys:     
                generate_video(join(args.input, scene, f'{args.model}/render/{key}/*.jpg'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/renbody')
    parser.add_argument('--scenes', type=str, nargs='+', default=['0013_06'])
    parser.add_argument('--model', type=str, default='smpl')
    parser.add_argument('--subs', type=str, nargs='+', default=[])
    parser.add_argument('--video', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)