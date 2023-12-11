import os
import cv2
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import join
from termcolor import colored

# fmt: off
import sys
sys.path.append('.')
from easyvolcap.utils.parallel_utils import parallel_execution
from easyvolcap.utils.data_utils import load_image, load_mask, list_to_tensor
# fmt: on

parser = argparse.ArgumentParser('This scripts will read all images from some camera and concatenate the masked images to output an video for inspection')
parser.add_argument('--data_root', default='/nas/home/xuzhen/datasets/xuzhen36/talk')
parser.add_argument('--image_dir', default='images')
parser.add_argument('--mask_dir', default='rvm')
parser.add_argument('--no_visualize', action='store_false', dest='visualize')
parser.add_argument('--no_store', action='store_false', dest='store')
parser.add_argument('--output', default='masked.mp4')
parser.add_argument('--fps', default=30.0, type=float)
parser.add_argument('--ratio', default=1.0, type=float)
parser.add_argument('--max_length', default=int(1e9))
args = parser.parse_args()

camera_names = sorted(listdir(join(args.data_root, args.image_dir)))
n_cameras = len(camera_names)  # this should stay fixed for all frames

image_names = sorted(listdir(join(args.data_root, args.image_dir, camera_names[0])))  # asserting first camera exists
image_list_list = [[join(args.data_root, args.image_dir, camera_name, image_name) for camera_name in camera_names] for image_name in image_names]  # create a list of list of image paths
mask_list_list = [[join(args.data_root, args.mask_dir, camera_name, image_name.replace('.jpg', '.png')) for camera_name in camera_names] for image_name in image_names]  # create a list of list of image paths
if not os.path.exists(mask_list_list[0][0]):
    mask_list_list = [[m.replace('.png', '.jpg') for m in ml] for ml in mask_list_list]

for frame_idx in tqdm(range(min(args.max_length, len(image_names))), desc='Processing frames'):
    image_list, mask_list = image_list_list[frame_idx], mask_list_list[frame_idx]

    # print('loading images')
    images = list_to_tensor(parallel_execution(image_list, args.ratio, action=load_image), 'cpu')
    # print('loading masks')
    masks = list_to_tensor(parallel_execution(mask_list, args.ratio, action=load_mask), 'cpu')

    # print('applying mask')
    images = images * masks  # the actual computation
    if "H_count" not in locals().keys() and 'H_count' not in globals().keys():
        B, C, H, W = images.shape  # H, W will be used later, this is messy, kind of...
        # constraints: H_count * W_count > n_cameras
        # H_count * H ~= W_count * W
        # H_count / W_count = W / H
        # W_count = H / W * H_count
        # H_count ** 2 * H/W > n_cameras
        H_count = int(np.sqrt(n_cameras / (H / W)))
        W_count = int(np.ceil(n_cameras / H_count))

    if args.store and 'out' not in locals().keys():
        output_path = join(args.data_root, args.output)
        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{W_count * W}x{H_count * H}',  # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', str(args.fps),  # frames per second
            '-i', '-',  # The input comes from a pipe
            '-crf', '24',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path),
        ]
        tqdm.write(colored(f'will write masked images video to: {colored(output_path, "blue")}', 'yellow'))
        tqdm.write(colored(f'with command: {colored(" ".join(command), "blue")}', 'yellow'))
        out = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=2**30)

    # print('filling canvas')
    canvas = images.new_empty(C, H_count * H, W_count * W)
    for i in range(H_count):
        for j in range(W_count):
            idx = i * H_count + j
            if idx >= B:
                canvas[:, i * H:(i + 1) * H, j * W:(j + 1) * W] = 0
            else:
                canvas[:, i * H:(i + 1) * H, j * W:(j + 1) * W] = images[idx]

    canvas: np.ndarray = (canvas.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # C, H, W -> H, W, C

    # print('storing or displaying')
    if args.store:
        out.stdin.write(canvas.tobytes())

    if args.visualize:
        cv2.imshow('masked images', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if args.store:
    out.stdin.close()
    out.stderr.close()
    out.wait()

"""
dirs=(my_313 my_315 my_377 my_386 my_387 my_390 my_392 my_393 my_394)
for dir in "${dirs[@]}"; do
    python scripts/visualize_masked_images.py --store --no_visualize --image_dir normal --mask_dir mask --fps 25 --ratio 0.25 --data_root data/my_zjumocap/$dir --output normal_$dir.mp4
done
"""
