import os
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='data/my_zjumocap/my_313')
parser.add_argument('--image_dir', default='images')
parser.add_argument('--mask_dir', default='mask')
parser.add_argument('--image_output', default='image_list.txt')
parser.add_argument('--mask_output', default='mask_list.txt')
args = parser.parse_args()

image_dir_path = os.path.join(args.input_dir, args.image_dir)
image_paths = glob.glob(f'{image_dir_path}/**/*.jpg') + glob.glob(f'{image_dir_path}/**/*.png')
image_paths = sorted(image_paths)
mask_paths = [img.replace(args.image_dir, args.mask_dir).replace('.jpg', '.png') for img in image_paths]

with open(os.path.join(args.input_dir, args.image_output), 'w') as f:
    for line in image_paths:
        f.write(line + '\n')

with open(os.path.join(args.input_dir, args.mask_output), 'w') as f:
    for line in mask_paths:
        f.write(line + '\n')
